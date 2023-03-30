// Copyright 2022 Alibaba
#include <torch/extension.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <iostream>
#include "cuda_util.cuh"
#include "render_util.cuh"
#include "data_spec_packed.cuh"
#include "cubemap_util.cuh"

namespace {

const int COLOR_CUDA_THREADS = 128;
const int COLOR_MIN_BLOCKS_PER_SM = 1;
const float PAI4 = 3.141592654 * 4;

namespace device {

//
__device__ __inline__ float trace_ray_weight(
        const PackedSparseGridSpec& __restrict__ grid,
        const RenderOptions& __restrict__ opt,
        SingleRaySpec& __restrict__ ray) {
    if (ray.tmin > ray.tmax) {
        return 0.f;
    }

    float weight = 0;
    float log_transmit = 0.f;
    float t = ray.tmin;

    while (t <= ray.tmax) {
#pragma unroll 3
        for (int j = 0; j < 3; ++j) {
            ray.pos[j] = fmaf(t, ray.dir[j], ray.origin[j]);
            ray.pos[j] = min(max(ray.pos[j], 0.f), grid.size[j] - 1.f);
            ray.l[j] = min(static_cast<int32_t>(ray.pos[j]), grid.size[j] - 2);
            ray.pos[j] -= static_cast<float>(ray.l[j]);
        }

        const float skip = compute_skip_dist(ray,
                       grid.links, grid.stride_x,
                       grid.size[2], 0);

        if (skip >= opt.step_size) {
            // For consistency, we skip the by step size
            t += ceilf(skip / opt.step_size) * opt.step_size;
            continue;
        }
        float sigma = trilerp_cuvol_one(
                grid.links, grid.density_data,
                grid.stride_x,
                grid.size[2],
                1,
                ray.l, ray.pos,
                0);
        if (opt.last_sample_opaque && t + opt.step_size > ray.tmax) {
            ray.world_step = 1e9;
        }

        if (sigma > opt.sigma_thresh) {
            const float pcnt = ray.world_step * sigma;
            log_transmit -= pcnt;

            if (_EXP(log_transmit) < opt.stop_thresh) {
                log_transmit = -1e3f;
                break;
            }
        }
        t += opt.step_size;
    }
    weight = _EXP(log_transmit);
    return weight;
}


__launch_bounds__(COLOR_CUDA_THREADS, COLOR_MIN_BLOCKS_PER_SM)
__global__ void estimate_color_metric_kernel(
        PackedSparseGridSpec grid,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> c2ws,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> w2cs,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> intrinsics,
        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> images,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> results,
        torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> store_color_weight,
        torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> store_color_res
        ) {
    // Now only support spherical harmonic basis.
    const int32_t block_id = blockIdx.x;
    const int32_t voxel_id = grid.links_inverse[block_id];
    const int32_t cam_id = threadIdx.x;
    const int32_t cam_num = c2ws.size(0);
    const int32_t basis_dim = grid.basis_dim;

    float basis_value[16] = {0.f};
    float color_observed[3] = {0.f};
    float color_weight = 0.f;

    typedef cub::BlockReduce<float, COLOR_CUDA_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // For each camera, calculate the basis_value, observed color, color weight.
    // grid coordinate
    const int32_t voxel_k = voxel_id % grid.size[2];
    const int32_t voxel_ij = voxel_id / grid.size[2];
    const int32_t voxel_j = voxel_ij % grid.size[1];
    const int32_t voxel_i = voxel_ij / grid.size[1];
    float voxel_pos_grid[3];
    voxel_pos_grid[0] = float(voxel_i);
    voxel_pos_grid[1] = float(voxel_j);
    voxel_pos_grid[2] = float(voxel_k);
    // world coordinate for 360 degree scenes
    // or ndc for forward facing scenes
    float voxel_pos_world[3];
    voxel_pos_world[0] = voxel_pos_grid[0] * grid._scaling_grid2world[0] + grid._offset_grid2world[0];
    voxel_pos_world[1] = voxel_pos_grid[1] * grid._scaling_grid2world[1] + grid._offset_grid2world[1];
    voxel_pos_world[2] = voxel_pos_grid[2] * grid._scaling_grid2world[2] + grid._offset_grid2world[2];

    if (intrinsics.size(1) >= 9){
        // forward facing scenes
        // calculate the world coordinate from the ndc
        voxel_pos_world[2] = (2.f * intrinsics[cam_id][8]) / (1.f - voxel_pos_world[2]);
        voxel_pos_world[0] *= voxel_pos_world[2] / intrinsics[cam_id][6];
        voxel_pos_world[1] *= voxel_pos_world[2] / intrinsics[cam_id][7];
    }

    // Calculate the ray.
    SingleRaySpec ray;
    // ray in the world coordinate
    ray.origin[0] = c2ws[cam_id][0][3];
    ray.origin[1] = c2ws[cam_id][1][3];
    ray.origin[2] = c2ws[cam_id][2][3];
    ray.dir[0] = voxel_pos_world[0] - ray.origin[0];
    ray.dir[1] = voxel_pos_world[1] - ray.origin[1];
    ray.dir[2] = voxel_pos_world[2] - ray.origin[2];

    if (intrinsics.size(1) >= 9){
        // forward facing scenes
        // calculate the rays in the ndc from the rays in the world coordinate
        const float ndc_coeff_0 = intrinsics[cam_id][6];
        const float ndc_coeff_1 = intrinsics[cam_id][7];
        const float near = intrinsics[cam_id][8];

        const float shift_to_near = (near - ray.origin[2]) / ray.dir[2];
    #pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            ray.origin[i] = fmaf(shift_to_near, ray.dir[i], ray.origin[i]);
        }
        ray.dir[0] = ndc_coeff_0 * (ray.dir[0] / ray.dir[2] - ray.origin[0] / ray.origin[2]);
        ray.dir[1] = ndc_coeff_1 * (ray.dir[1] / ray.dir[2] - ray.origin[1] / ray.origin[2]);
        ray.dir[2] = 2.f * near / ray.origin[2];

        ray.origin[0] = ndc_coeff_0 * (ray.origin[0] / ray.origin[2]);
        ray.origin[1] = ndc_coeff_1 * (ray.origin[1] / ray.origin[2]);
        ray.origin[2] = 1.f - ray.dir[2];
    }

    _normalize(ray.dir);

    // Calculate the basis value using normalized ray directions in the ndc coordinate.
    calc_sh(basis_dim, ray.dir, basis_value);

    // Project the voxel onto the camera's image plane.
    float projected_x = w2cs[cam_id][0][0] * voxel_pos_world[0] + w2cs[cam_id][0][1] * voxel_pos_world[1]
                        + w2cs[cam_id][0][2] * voxel_pos_world[2] + w2cs[cam_id][0][3];
    float projected_y = w2cs[cam_id][1][0] * voxel_pos_world[0] + w2cs[cam_id][1][1] * voxel_pos_world[1]
                        + w2cs[cam_id][1][2] * voxel_pos_world[2] + w2cs[cam_id][1][3];
    float projected_z = w2cs[cam_id][2][0] * voxel_pos_world[0] + w2cs[cam_id][2][1] * voxel_pos_world[1]
                        + w2cs[cam_id][2][2] * voxel_pos_world[2] + w2cs[cam_id][2][3];
    float img_x = (projected_x / projected_z) * intrinsics[cam_id][0] + intrinsics[cam_id][2];
    float img_y = (projected_y / projected_z) * intrinsics[cam_id][1] + intrinsics[cam_id][3];

    if (projected_z > 0.01f && img_x > 1.f && img_x < intrinsics[cam_id][4] - 2.f
        && img_y > 1.f && img_y < intrinsics[cam_id][5] - 2.f) {
        // Inside image plane.
        // Calculate the observed color.
        const int img_x_left = (int)img_x;
        const int img_x_right = img_x + 1;
        const int img_y_top = (int)img_y;
        const int img_y_bottom = img_y + 1;
        img_x -= float(img_x_left);
        img_y -= float(img_y_top);
        // f(0,0)(1-x)(1-y)+f(1,0)x(1-y)+f(0,1)(1-x)y+f(1,1)xy
        #pragma unroll 3
        for (int i = 0; i < 3; ++i)
            color_observed[i] =
                images[cam_id][img_y_top][img_x_left][i] * (1.f - img_y) * (1.f - img_x)
                + images[cam_id][img_y_top][img_x_right][i] * (1.f - img_y) * img_x
                + images[cam_id][img_y_bottom][img_x_left][i] * img_y * (1.f - img_x)
                + images[cam_id][img_y_bottom][img_x_right][i] * img_y * img_x
                - 0.5f;

        // Calculate the weight along the ray.
        // - Calculate tmin and tmax of the ray.
        // -- Warning: modifies ray.origin to grid coordinate.
        transform_coord(ray.origin, grid._scaling, grid._offset);
        // -- Warning: modifies ray.dir
        ray.world_step = _get_delta_scale(grid._scaling, ray.dir) * opt.step_size;
        ray.tmin = opt.near_clip / ray.world_step * opt.step_size;
        ray.tmax = 2e3f;
        for (int i = 0; i < 3; ++i) {
            const float invdir = 1.f / ray.dir[i];
            const float t1 = (-0.5f - ray.origin[i]) * invdir;
            const float t2 = (grid.size[i] - 0.5f  - ray.origin[i]) * invdir;
            const float t3 = (voxel_pos_grid[i] - ray.origin[i]) * invdir;
            if (ray.dir[i] != 0.f) {
                ray.tmin = max(ray.tmin, min(t1, t2));
                ray.tmax = min(ray.tmax, t3);
            }
        }

        color_weight = trace_ray_weight(grid, opt, ray);
    }

//     __syncthreads();

    float weight_sum = BlockReduce(temp_storage).Sum(color_weight, cam_num);
    float weight_sum_as_den = weight_sum + 1e-9f;
    float scale_parameter = PAI4 / weight_sum_as_den;
    float weighted_basis_color = 0.f;               // T_q * c_q * Y_l(q)
    float weighted_basis_color_sum = 0.f;           // sum_{q=1}^V {T_q * c_q * Y(q)}
    float record_weighted_basis = 0.f;              // T_q * Y_l(q)
    // Estimate each spherical harmonic coefficient in turn.
    for (size_t idx_sh = 0; idx_sh < basis_dim; ++idx_sh) {
        // weighted_basis = color_weight * basis_value[idx_sh];
        record_weighted_basis = color_weight * basis_value[idx_sh];
        for (int i = 0; i < 3; ++i) {
            weighted_basis_color = record_weighted_basis * color_observed[i];
            __syncthreads();  // block-wide sync barrier to re-use shared mem safely
            weighted_basis_color_sum = BlockReduce(temp_storage).Sum(weighted_basis_color, cam_num);
            if (cam_id == 0)
                results[block_id][idx_sh + basis_dim * i] = scale_parameter * weighted_basis_color_sum;
            __syncthreads();  // sync to make new global memory value available to all threads
            color_observed[i] -= results[block_id][idx_sh + basis_dim * i] * basis_value[idx_sh];
        }
    }

    float weighted_color_res = color_weight * (
            color_observed[0] * color_observed[0] +
            color_observed[1] * color_observed[1] +
            color_observed[2] * color_observed[2]
          );
    __syncthreads();  // block-wide sync barrier to re-use shared mem safely
    float weighted_color_res_sum = BlockReduce(temp_storage).Sum(weighted_color_res, cam_num);
    if (cam_id == 0) {
        store_color_res[block_id] = weighted_color_res_sum;
        store_color_weight[block_id] = weight_sum;
    }

}

// View-dependent version
// When calculating the residual color of the image i,
// the image i is not used for SH estimation
__launch_bounds__(COLOR_CUDA_THREADS, COLOR_MIN_BLOCKS_PER_SM)
__global__ void vd_estimate_color_metric_kernel(
        PackedSparseGridSpec grid,
        RenderOptions opt,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> c2ws,
        torch::PackedTensorAccessor32<float, 3, torch::RestrictPtrTraits> w2cs,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> intrinsics,
        torch::PackedTensorAccessor32<float, 4, torch::RestrictPtrTraits> images,
        torch::PackedTensorAccessor32<float, 2, torch::RestrictPtrTraits> results,
        torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> store_color_weight,
        torch::PackedTensorAccessor32<float, 1, torch::RestrictPtrTraits> store_color_res
        ) {
    // Now only support spherical harmonic basis.
    const int32_t block_id = blockIdx.x;
    const int32_t voxel_id = grid.links_inverse[block_id];
    const int32_t cam_id = threadIdx.x;
    const int32_t cam_num = c2ws.size(0);
    const int32_t basis_dim = grid.basis_dim;

    float basis_value[16] = {0.f};
//     float color_observed[3] = {0.f};
    float color_weight = 0.f;

    __shared__ float color_observed[COLOR_CUDA_THREADS][3];
    #pragma unroll 3
    for (int i = 0; i < 3; ++i) {
        color_observed[cam_id][i] = 0.f;
    }

    typedef cub::BlockReduce<float, COLOR_CUDA_THREADS> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;

    // For each camera, calculate the basis_value, observed color, color weight.
    // grid coordinate
    const int32_t voxel_k = voxel_id % grid.size[2];
    const int32_t voxel_ij = voxel_id / grid.size[2];
    const int32_t voxel_j = voxel_ij % grid.size[1];
    const int32_t voxel_i = voxel_ij / grid.size[1];
    float voxel_pos_grid[3];
    voxel_pos_grid[0] = float(voxel_i);
    voxel_pos_grid[1] = float(voxel_j);
    voxel_pos_grid[2] = float(voxel_k);
    // world coordinate for 360 degree scenes
    // or ndc for forward facing scenes
    float voxel_pos_world[3];
    voxel_pos_world[0] = voxel_pos_grid[0] * grid._scaling_grid2world[0] + grid._offset_grid2world[0];
    voxel_pos_world[1] = voxel_pos_grid[1] * grid._scaling_grid2world[1] + grid._offset_grid2world[1];
    voxel_pos_world[2] = voxel_pos_grid[2] * grid._scaling_grid2world[2] + grid._offset_grid2world[2];

    if (intrinsics.size(1) >= 9){
        // forward facing scenes
        // calculate the world coordinate from the ndc
        voxel_pos_world[2] = (2.f * intrinsics[cam_id][8]) / (1.f - voxel_pos_world[2]);
        voxel_pos_world[0] *= voxel_pos_world[2] / intrinsics[cam_id][6];
        voxel_pos_world[1] *= voxel_pos_world[2] / intrinsics[cam_id][7];
    }

    // Calculate the ray.
    SingleRaySpec ray;
    // ray in the world coordinate
    ray.origin[0] = c2ws[cam_id][0][3];
    ray.origin[1] = c2ws[cam_id][1][3];
    ray.origin[2] = c2ws[cam_id][2][3];
    ray.dir[0] = voxel_pos_world[0] - ray.origin[0];
    ray.dir[1] = voxel_pos_world[1] - ray.origin[1];
    ray.dir[2] = voxel_pos_world[2] - ray.origin[2];

    if (intrinsics.size(1) >= 9){
        // forward facing scenes
        // calculate the rays in the ndc from the rays in the world coordinate
        const float ndc_coeff_0 = intrinsics[cam_id][6];
        const float ndc_coeff_1 = intrinsics[cam_id][7];
        const float near = intrinsics[cam_id][8];

        const float shift_to_near = (near - ray.origin[2]) / ray.dir[2];
    #pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            ray.origin[i] = fmaf(shift_to_near, ray.dir[i], ray.origin[i]);
        }
        ray.dir[0] = ndc_coeff_0 * (ray.dir[0] / ray.dir[2] - ray.origin[0] / ray.origin[2]);
        ray.dir[1] = ndc_coeff_1 * (ray.dir[1] / ray.dir[2] - ray.origin[1] / ray.origin[2]);
        ray.dir[2] = 2.f * near / ray.origin[2];

        ray.origin[0] = ndc_coeff_0 * (ray.origin[0] / ray.origin[2]);
        ray.origin[1] = ndc_coeff_1 * (ray.origin[1] / ray.origin[2]);
        ray.origin[2] = 1.f - ray.dir[2];
    }

    _normalize(ray.dir);

    // Calculate the basis value using normalized ray directions in the ndc coordinate.
    calc_sh(basis_dim, ray.dir, basis_value);

    // Project the voxel onto the camera's image plane.
    float projected_x = w2cs[cam_id][0][0] * voxel_pos_world[0] + w2cs[cam_id][0][1] * voxel_pos_world[1]
                        + w2cs[cam_id][0][2] * voxel_pos_world[2] + w2cs[cam_id][0][3];
    float projected_y = w2cs[cam_id][1][0] * voxel_pos_world[0] + w2cs[cam_id][1][1] * voxel_pos_world[1]
                        + w2cs[cam_id][1][2] * voxel_pos_world[2] + w2cs[cam_id][1][3];
    float projected_z = w2cs[cam_id][2][0] * voxel_pos_world[0] + w2cs[cam_id][2][1] * voxel_pos_world[1]
                        + w2cs[cam_id][2][2] * voxel_pos_world[2] + w2cs[cam_id][2][3];
    float img_x = (projected_x / projected_z) * intrinsics[cam_id][0] + intrinsics[cam_id][2];
    float img_y = (projected_y / projected_z) * intrinsics[cam_id][1] + intrinsics[cam_id][3];

    if (projected_z > 0.01f && img_x > 1.f && img_x < intrinsics[cam_id][4] - 2.f
        && img_y > 1.f && img_y < intrinsics[cam_id][5] - 2.f) {
        // Inside image plane.
        // Calculate the observed color.
        const int img_x_left = (int)img_x;
        const int img_x_right = img_x + 1;
        const int img_y_top = (int)img_y;
        const int img_y_bottom = img_y + 1;
        img_x -= float(img_x_left);
        img_y -= float(img_y_top);
        // f(0,0)(1-x)(1-y)+f(1,0)x(1-y)+f(0,1)(1-x)y+f(1,1)xy
        #pragma unroll 3
        for (int i = 0; i < 3; ++i)
            color_observed[cam_id][i] =
                images[cam_id][img_y_top][img_x_left][i] * (1.f - img_y) * (1.f - img_x)
                + images[cam_id][img_y_top][img_x_right][i] * (1.f - img_y) * img_x
                + images[cam_id][img_y_bottom][img_x_left][i] * img_y * (1.f - img_x)
                + images[cam_id][img_y_bottom][img_x_right][i] * img_y * img_x
                - 0.5f;

        // Calculate the weight along the ray.
        // - Calculate tmin and tmax of the ray.
        // -- Warning: modifies ray.origin to grid coordinate.
        transform_coord(ray.origin, grid._scaling, grid._offset);
        // -- Warning: modifies ray.dir
        ray.world_step = _get_delta_scale(grid._scaling, ray.dir) * opt.step_size;
        ray.tmin = opt.near_clip / ray.world_step * opt.step_size;
        ray.tmax = 2e3f;
        for (int i = 0; i < 3; ++i) {
            const float invdir = 1.f / ray.dir[i];
            const float t1 = (-0.5f - ray.origin[i]) * invdir;
            const float t2 = (grid.size[i] - 0.5f  - ray.origin[i]) * invdir;
            const float t3 = (voxel_pos_grid[i] - ray.origin[i]) * invdir;
            if (ray.dir[i] != 0.f) {
                ray.tmin = max(ray.tmin, min(t1, t2));
                ray.tmax = min(ray.tmax, t3);
            }
        }

        color_weight = trace_ray_weight(grid, opt, ray);
    }


//  The return value is undefined in threads other than thread0.
    float weight_sum = BlockReduce(temp_storage).Sum(color_weight, cam_num);
    float weighted_basis_color = 0.f;
    float weighted_basis_color_sum = 0.f;

    // view dependent (vd)
    __shared__ float shared_weight_sum;
    __shared__ float shared_weighted_basis_color_sum;
    __shared__ float vd_scale_parameter[COLOR_CUDA_THREADS];
    if (cam_id == 0) {
        // store `weight_sum` in shared memory so that it can be available to all threads
        shared_weight_sum = weight_sum;
    }
    __syncthreads();  // sync to make new shared memory value available to all threads
    vd_scale_parameter[cam_id] = PAI4 / (shared_weight_sum - color_weight + 1e-9f);
//     __syncthreads();

    float vd_color_res[COLOR_CUDA_THREADS][3];
    float record_weighted_basis[16] = {0.f};
    #pragma unroll
    for (size_t idx_sh = 0; idx_sh < basis_dim; ++idx_sh) {
        record_weighted_basis[idx_sh] = color_weight * basis_value[idx_sh];
    }
    float vd_weighted_color_res = 0.f;
    // Estimate each spherical harmonic coefficient in turn.
    for (size_t idx_view = 0; idx_view < cam_num; ++idx_view) {
        // initialize view dependent residual color
        for (size_t j = 0; j < cam_num; ++j) {
            #pragma unroll 3
            for (size_t i = 0; i < 3; ++i) {
                vd_color_res[j][i] = color_observed[j][i];
            }
        }
        // calculate view dependent residual color
        for (size_t idx_sh = 0; idx_sh < basis_dim; ++idx_sh) {
            for (int i = 0; i < 3; ++i) {
                weighted_basis_color = record_weighted_basis[idx_sh] * vd_color_res[cam_id][i];
                __syncthreads();  // block-wide sync barrier to re-use shared mem safely
                weighted_basis_color_sum = BlockReduce(temp_storage).Sum(weighted_basis_color, cam_num);
                if (cam_id == 0) shared_weighted_basis_color_sum = weighted_basis_color_sum;
                __syncthreads();  // sync to make new shared memory value available to all threads
                if (cam_id == idx_view) shared_weighted_basis_color_sum -= weighted_basis_color;
                __syncthreads();  // sync to make new shared memory value available to all threads
                vd_color_res[cam_id][i] -= vd_scale_parameter[idx_view] * shared_weighted_basis_color_sum * basis_value[idx_sh];
            }
        }

        if (idx_view == cam_id) {
            vd_weighted_color_res = color_weight * (
                vd_color_res[cam_id][0] * vd_color_res[cam_id][0] +
                vd_color_res[cam_id][1] * vd_color_res[cam_id][1] +
                vd_color_res[cam_id][2] * vd_color_res[cam_id][2]
            );
        }

    }

    __syncthreads();  // block-wide sync barrier to re-use shared mem safely
    float vd_weighted_color_res_sum = BlockReduce(temp_storage).Sum(vd_weighted_color_res, cam_num);
    if (cam_id == 0) {
        store_color_res[block_id] = vd_weighted_color_res_sum;
        store_color_weight[block_id] = weight_sum;
    }

}


}  // namespace device
}  // namespace

torch::Tensor estimate_color_metric(
        SparseGridSpec & grid, RenderOptions& opt,
        Tensor c2ws, Tensor w2cs, Tensor intrinsics, Tensor images,
        Tensor & store_color_weight,
        Tensor & store_color_res) {
    DEVICE_GUARD(grid.density_data);
    grid.check();
    CHECK_INPUT(c2ws);
    CHECK_INPUT(w2cs);
    CHECK_INPUT(intrinsics);
    CHECK_INPUT(images);
    CHECK_INPUT(store_color_weight);
    CHECK_INPUT(store_color_res);

    auto options =
        torch::TensorOptions()
        .dtype(grid.density_data.dtype())
        .layout(torch::kStrided)
        .device(grid.density_data.device())
        .requires_grad(false);
    torch::Tensor results = torch::empty({grid.density_data.size(0), grid.basis_dim * 3}, options);

    if(w2cs.size(0) > COLOR_CUDA_THREADS) {
        std::cout << "[Error] cameras are more than cuda threads: "
                  << w2cs.size(0) << " vs " << COLOR_CUDA_THREADS << std::endl;
        return results;
    }

    const int32_t blocks = grid.density_data.size(0);
    const int32_t num_threads_per_block = w2cs.size(0);
    device::estimate_color_metric_kernel<<<blocks, num_threads_per_block>>>(
            grid,
            opt,
            c2ws.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            w2cs.packed_accessor32<float, 3, torch::RestrictPtrTraits>(),
            intrinsics.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            images.packed_accessor32<float, 4, torch::RestrictPtrTraits>(),
            results.packed_accessor32<float, 2, torch::RestrictPtrTraits>(),
            store_color_weight.packed_accessor32<float, 1, torch::RestrictPtrTraits>(),
            store_color_res.packed_accessor32<float, 1, torch::RestrictPtrTraits>()
            );

    return results;
}
