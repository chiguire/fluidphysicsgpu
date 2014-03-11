/*** FUNCTIONS FOR FLOAT 2 ***/

__kernel void add_source_float2(__global float2* prev,
                                __global float2* curr,
                                int data_width,
                                float dt) {
  uint global_addr_x, global_addr_y, idx;
  global_addr_x = get_global_id(0);
  global_addr_y = get_global_id(1);
  idx = (global_addr_y)*(data_width)+global_addr_x;
  curr[idx] += dt*prev[idx];

}
__kernel void lin_solve_float2(__global float2* prev,
                               __global float2* curr,
                               int data_width,
                               float a,
                               float c) {

  uint global_addr_x, global_addr_y, idx11, idx01, idx21, idx10, idx12;
  global_addr_x = get_global_id(0)+1;
  global_addr_y = get_global_id(1)+1;
  idx11 = (global_addr_y)*(data_width)+(global_addr_x);
  idx01 = (global_addr_y)*(data_width)+(global_addr_x-1);
  idx21 = (global_addr_y)*(data_width)+(global_addr_x+1);
  idx10 = (global_addr_y-1)*(data_width)+(global_addr_x);
  idx12 = (global_addr_y+1)*(data_width)+(global_addr_x);

  curr[idx11] = (prev[idx11] + a*(curr[idx01]+curr[idx21]+curr[idx10]+curr[idx12]))/c;
}

/* Call with one dimension work item, equals to fluid width */
__kernel void set_bnd_float2(__global float2 *arr,
                             int data_width) {
  uint global_addr_x, global_addr_y, 
       idx0I, idxN1I, idxI0, idxIN1,
       idx1I, idxNI,  idxI1, idxIN;
  global_addr_x = get_global_id(0)+1;
  idx0I = (global_addr_x)*(data_width)+0;
  idxN1I = (global_addr_x)*(data_width)+data_width-1;
  idxI0 = 0*(data_width)+(global_addr_x);
  idxIN1 = (data_width-1)*(data_width)+(global_addr_x);

  idx1I = (global_addr_x)*(data_width)+1;
  idxNI = (global_addr_x)*(data_width)+(data_width-2);
  idxI1 = 1*(data_width)+(global_addr_x);
  idxIN = (data_width-2)*(data_width)+(global_addr_x);

  arr[idx0I] = arr[idx1I]*(float2)(-1,1);
  arr[idxN1I] = arr[idxNI]*(float2)(-1,1);
  arr[idxI0] = arr[idxI1]*(float2)(1,-1);
  arr[idxI0] = arr[idxIN]*(float2)(1,-1);
}

__kernel void set_bnd_float2_end(__global float2 *arr,
                                 int data_width) {
  uint idx00, idx10, idx01, idx0N1, idx1N1, idx0N,
       idxN10, idxN0, idxN11, idxN1N1, idxNN1, idxN1N;

  idx00   = 0;
  idx10   = 1;
  idx01   = data_width;
  idx0N1  = (data_width - 1) * data_width + 0;
  idx1N1  = (data_width - 1) * data_width + 1;
  idx0N   = (data_width - 2) * data_width;
  idxN10  = data_width - 1;
  idxN0   = data_width - 2;
  idxN11  = (data_width * 1) + data_width - 1;
  idxN1N1 = (data_width - 1) * data_width + (data_width - 1);
  idxNN1  = (data_width - 1) * data_width + (data_width - 2);
  idxN1N  = (data_width - 2) * data_width + (data_width - 1);

  arr[idx00]   = 0.5f * (arr[idx10]  + arr[idx01]);
  arr[idx0N1]  = 0.5f * (arr[idx1N1] + arr[idx0N]);
  arr[idxN10]  = 0.5f * (arr[idxN0]  + arr[idxN11]);
  arr[idxN1N1] = 0.5f * (arr[idxNN1] + arr[idxN1N]);
}

/*** FUNCTIONS FOR FLOAT 1 ***/

__kernel void lin_solve_float(__global float* prev,
                              __global float* curr,
                              int data_width,
                              float a,
                              float c) {

  uint global_addr_x, global_addr_y, idx11, idx01, idx21, idx10, idx12;
  global_addr_x = get_global_id(0)+1;
  global_addr_y = get_global_id(1)+1;
  idx11 = (global_addr_y)*(data_width)+(global_addr_x);
  idx01 = (global_addr_y)*(data_width)+(global_addr_x-1);
  idx21 = (global_addr_y)*(data_width)+(global_addr_x+1);
  idx10 = (global_addr_y-1)*(data_width)+(global_addr_x);
  idx12 = (global_addr_y+1)*(data_width)+(global_addr_x);
  curr[idx11] = (prev[idx11] + a*(curr[idx01]+curr[idx21]+curr[idx10]+curr[idx12]))/c;
}

/* Call with one dimension work item, equals to fluid width */
__kernel void set_bnd_float(__global float *arr,
                            int data_width) {
  uint global_addr_x, global_addr_y, 
       idx0I, idxN1I, idxI0, idxIN1,
       idx1I, idxNI,  idxI1, idxIN;
  global_addr_x = get_global_id(0)+1;
  idx0I = (global_addr_x)*(data_width)+0;
  idxN1I = (global_addr_x)*(data_width)+data_width-1;
  idxI0 = 0*(data_width)+(global_addr_x);
  idxIN1 = (data_width-1)*(data_width)+(global_addr_x);

  idx1I = (global_addr_x)*(data_width)+1;
  idxNI = (global_addr_x)*(data_width)+(data_width-2);
  idxI1 = 1*(data_width)+(global_addr_x);
  idxIN = (data_width-2)*(data_width)+(global_addr_x);

  arr[idx0I] = arr[idx1I];
  arr[idxN1I] = arr[idxNI];
  arr[idxI0] = arr[idxI1];
  arr[idxI0] = arr[idxIN];
}

__kernel void set_bnd_float_end(__global float *arr,
                                int data_width) {
  uint idx00, idx10, idx01, idx0N1, idx1N1, idx0N,
       idxN10, idxN0, idxN11, idxN1N1, idxNN1, idxN1N;

  idx00   = 0;
  idx10   = 1;
  idx01   = data_width;
  idx0N1  = (data_width - 1) * data_width + 0;
  idx1N1  = (data_width - 1) * data_width + 1;
  idx0N   = (data_width - 2) * data_width;
  idxN10  = data_width - 1;
  idxN0   = data_width - 2;
  idxN11  = (data_width * 1) + data_width - 1;
  idxN1N1 = (data_width - 1) * data_width + (data_width - 1);
  idxNN1  = (data_width - 1) * data_width + (data_width - 2);
  idxN1N  = (data_width - 2) * data_width + (data_width - 1);

  arr[idx00]   = 0.5f * (arr[idx10] + arr[idx01]);
  arr[idx0N1]  = 0.5f * (arr[idx1N1] + arr[idx0N]);
  arr[idxN10]  = 0.5f * (arr[idxN0] + arr[idxN11]);
  arr[idxN1N1] = 0.5f * (arr[idxNN1] + arr[idxN1N]);
}

/** PROJECT **/
__kernel void project_start(__global float2 *uv,
                            __global float *p0,
                            __global float *p1,
                            int data_width) {
  uint global_addr_x, global_addr_y, idx11, idx01, idx21, idx10, idx12;
  global_addr_x = get_global_id(0)+1;
  global_addr_y = get_global_id(1)+1;
  idx11 = (global_addr_y)*(data_width)+(global_addr_x);
  idx01 = (global_addr_y)*(data_width)+(global_addr_x-1);
  idx21 = (global_addr_y)*(data_width)+(global_addr_x+1);
  idx10 = (global_addr_y-1)*(data_width)+(global_addr_x);
  idx12 = (global_addr_y+1)*(data_width)+(global_addr_x);

  p0[idx11] = -0.5f*(uv[idx21].x-uv[idx01].x+uv[idx12].y-uv[idx10].y)/(data_width-2);
  p1[idx11] = 0;
}

__kernel void project_end(__global float2 *uv,
                          __global float *p,
                          int data_width) {
  uint global_addr_x, global_addr_y, idx11, idx01, idx21, idx10, idx12;
  global_addr_x = get_global_id(0)+1;
  global_addr_y = get_global_id(1)+1;
  idx11 = (global_addr_y)*(data_width)+(global_addr_x);
  idx01 = (global_addr_y)*(data_width)+(global_addr_x-1);
  idx21 = (global_addr_y)*(data_width)+(global_addr_x+1);
  idx10 = (global_addr_y-1)*(data_width)+(global_addr_x);
  idx12 = (global_addr_y+1)*(data_width)+(global_addr_x);

  uv[idx11] -= (float2)(0.5f*(data_width-2)*(p[idx21]-p[idx01]),
                        0.5f*(data_width-2)*(p[idx12]-p[idx10]));
}

/** ADVECT **/
__kernel void advect_float2(__global float2 *uv0,
                            __global float2 *uv1,
                            __global float2 *uv,
                            int data_width,
                            float dt0) {
  uint global_addr_x, global_addr_y, idx, idx00, idx01, idx10, idx11;
  float2 p, pf0, pf1;
  int2 pi0, pi1;
  global_addr_x = get_global_id(0)+1;
  global_addr_y = get_global_id(1)+1;
  idx = (global_addr_y)*(data_width)+(global_addr_x);
  p = clamp((float2)(global_addr_x - dt0*uv[idx].x, global_addr_y - dt0*uv[idx].y),
            (float2)(0.5f, 0.5f),
            (float2)((data_width-2)+0.5f, (data_width-2)+0.5f));
  pi0 = convert_int2(p);
  pi1 = pi0 + (float2)(1,1);
  pf1 = p-pi0;
  pf0 = 1-pf1;

  idx00 = pi0.y*data_width+pi0.x;
  idx01 = pi1.y*data_width+pi0.x;
  idx11 = pi1.y*data_width+pi1.x;
  idx10 = pi0.y*data_width+pi1.x;

  uv1[idx] = pf0.x * (pf0.y * uv0[idx00] + pf1.y * uv0[idx01]) +
             pf1.x * (pf0.y * uv0[idx10] + pf1.y * uv0[idx11]);
}

__kernel void advect_float(__global float *arr0,
                           __global float *arr1,
                           __global float2 *uv,
                           int data_width,
                           float dt0) {
  uint global_addr_x, global_addr_y, idx, idx00, idx01, idx10, idx11;
  float2 p, pf0, pf1;
  int2 pi0, pi1;
  global_addr_x = get_global_id(0)+1;
  global_addr_y = get_global_id(1)+1;
  idx = (global_addr_y)*(data_width)+(global_addr_x);
  p = clamp((float2)(global_addr_x - dt0*uv[idx].x, global_addr_y - dt0*uv[idx].y),
            (float2)(0.5f, 0.5f),
            (float2)((data_width-2)+0.5f, (data_width-2)+0.5f));
  pi0 = convert_int2(p);
  pi1 = pi0 + (float2)(1,1);
  pf1 = p-pi0;
  pf0 = 1-pf1;

  idx00 = pi0.y*data_width+pi0.x;
  idx01 = pi1.y*data_width+pi0.x;
  idx11 = pi1.y*data_width+pi1.x;
  idx10 = pi0.y*data_width+pi1.x;

  arr1[idx] = pf0.x * (pf0.y * arr0[idx00] + pf1.y * arr0[idx01]) +
              pf1.x * (pf0.y * arr0[idx10] + pf1.y * arr0[idx11]);
}