////////////////////////////////////////////////////////////////////////////////
//
// (C) Ciro Duran, Andy Thomason 2012, 2013, 2014
//
// Modular Framework for OpenGLES2 rendering on multiple platforms.
//
// 2D fluid dynamics on GPU
//
// Level: 2
//
// Demonstrates:
// Use of OpenCL for GPGPU
// Copied OpenCL initialisation code from
// http://www.drdobbs.com/parallel/a-gentle-introduction-to-opencl/231002854?pgno=3

namespace octet {
  class engine : public app {
    typedef mat4t mat4t;
    typedef vec4 vec4;

    // shaders to draw triangles
    fluid_shader fshader;
    color_shader cshader;

    // helper for drawing text
    //text_overlay overlay;

    // OpenCL stuff
    cl_device_id clDeviceID;
    cl_context clContext;
    cl_program clProgram;
    cl_kernel clAddSourceFloat2Kernel;
    cl_kernel clLinSolveFloat2Kernel;
    cl_kernel clSetBoundFloat2Kernel;
    cl_kernel clSetBoundEndFloat2Kernel;
    cl_kernel clAddSourceFloatKernel;
    cl_kernel clLinSolveFloatKernel;
    cl_kernel clSetBoundFloatKernel;
    cl_kernel clSetBoundEndFloatKernel;
    cl_kernel clProjectStartKernel;
    cl_kernel clProjectEndKernel;
    cl_kernel clAdvectFloat2Kernel;
    cl_kernel clAdvectFloatKernel;

    cl_command_queue clQueue;

    cl_mem uv0_buffer;
    cl_mem uv1_buffer;
    cl_mem dens0_buffer;
    cl_mem dens1_buffer;

    GLuint fluidPositionsVBO;
    GLuint fluidIndicesVBO;
    GLuint densVBO;

    // Fluid data
    unsigned int N;
    unsigned int Nborder;
    float dt;
    float diff;
    float visc;
    float force;
    float source;

    /*** OPENCL SPECIFIC FUNCTIONS ***/

    cl_device_id createDevice() {
      cl_platform_id platform;
      cl_device_id dev;
      int err;

      // Identify a platform
      err = clGetPlatformIDs(1, &platform, NULL);
      if (err < 0) {
        perror("Could not identify a platform");
        exit(1);
      }

      //Access a device
      err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &dev, NULL);
      if (err == CL_DEVICE_NOT_FOUND) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &dev, NULL);
        printf("Creating OpenCL CPU profile.\n");
      } else {
        printf("Creating OpenCL GPU profile.\n");
      }
      if (err < 0) {
        printf("Could not access any device");
        exit(1);
      }

      return dev;
    }

    cl_kernel createKernel(cl_program prg, const char *kernel_name) {
      cl_int err;
      cl_kernel k = clCreateKernel(clProgram, kernel_name, &err);
      if (err < 0) {
        printf("Could not create kernel %s", kernel_name);
        return NULL; 
      }
      return k;
    }

    void writeArray(cl_mem dst, float *src, unsigned int size) {
      cl_int err = clEnqueueWriteBuffer(clQueue, dst, CL_TRUE, 0, size*sizeof(float), (void *)src, 0, NULL, NULL);
      if (err < 0) {
        printf("Could not write array.");
      }
    }

    void readArray(cl_mem src, float *dst, unsigned int size) {
      cl_int err = clEnqueueReadBuffer(clQueue, src, CL_TRUE, 0, size*sizeof(float), (void *)dst, 0, NULL, NULL);
      if (err < 0) {
        printf("Could not read array.");
      }
    }

    cl_program buildProgram(cl_context ctx, cl_device_id dev, const char *filename) {
      
      cl_program program;
      FILE *program_handle;
      char *program_buffer, *program_log;
      size_t program_size, log_size;
      int err;

      // Read program file and place content into buffer
      program_handle = fopen(app_utils::get_path(filename), "r");
      if (program_handle == NULL) {
        perror("Could not find the program file");
        exit(1);
      }
      fseek(program_handle, 0, SEEK_END);
      program_size = ftell(program_handle);
      rewind(program_handle);
      program_buffer = (char *)malloc(program_size + 1);
      program_buffer[program_size] = '\0';
      fread(program_buffer, sizeof(char), program_size, program_handle);
      fclose(program_handle);

      // Create program from file
      program = clCreateProgramWithSource(ctx, 1, 
        (const char **)&program_buffer, &program_size, &err);
      if (err < 0) {
        perror("Could not create the program");
        exit(1);
      }
      free(program_buffer);

      // Build program
      err = clBuildProgram(program, 0, NULL, "-cl-nv-verbose", NULL, NULL);
      if (err < 0) {
        // Find size of log and print to std output
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
          0, NULL, &log_size);
        program_log = (char *) malloc(log_size + 1);
        program_log[log_size] = '\0';
        clGetProgramBuildInfo(program, dev, CL_PROGRAM_BUILD_LOG,
          log_size + 1, program_log, NULL);
        printf("%s\n", program_log);
        free(program_log);
        exit(1);
      }
      return program; 
    }

    void initOpenCL() {
      cl_int err;
      int size = (Nborder)*(Nborder);

      clDeviceID = createDevice();
      clContext = clCreateContext(NULL, 1, &clDeviceID, NULL, NULL, &err);
      if (err < 0) {
        perror("Could not create a context");
        return; //exit(1);
      }

      // Build program
      clProgram = buildProgram(clContext, clDeviceID, "assets/opencl/fluids.cl");

      // Create data buffer
      dynarray <float>uv0;
      dynarray <float>uv1;
      dynarray <float>dens0;
      dynarray <float>dens1;

      uv0.resize(size*2);
      uv1.resize(size*2);
      dens0.resize(size);
      dens1.resize(size);

      for (int i = 0; i != size*2; i++) {
        uv0[i] = 0;
        uv1[i] = 0;
      }
      
      for (int i = 0; i != size; i++) {
        dens0[i] = 0;
        dens1[i] = 0;
      }

      uv0_buffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, size * 2 * sizeof(float), uv0.data(), &err);
      dens0_buffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, size * sizeof(float), dens0.data(), &err);
      uv1_buffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, size * 2 * sizeof(float), uv1.data(), &err);
      dens1_buffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, size * sizeof(float), dens1.data(), &err);
      if (err < 0) {
        perror("Could not create a buffer");
        return; //exit(1);
      }
      
      // Create a Command Queue
      clQueue = clCreateCommandQueue(clContext, clDeviceID, 0, &err);
      if (err < 0) {
        perror("Could not create a command queue");
        return; //exit(1);
      }

      // Create a Kernel
      clAddSourceFloat2Kernel = createKernel(clProgram, "add_source_float2");
      clLinSolveFloat2Kernel = createKernel(clProgram, "lin_solve_float2");
      clSetBoundFloat2Kernel = createKernel(clProgram, "set_bnd_float2");
      clSetBoundEndFloat2Kernel = createKernel(clProgram, "set_bnd_float2_end");
      
      clAddSourceFloatKernel = createKernel(clProgram, "add_source_float");
      clLinSolveFloatKernel = createKernel(clProgram, "lin_solve_float");
      clSetBoundFloatKernel = createKernel(clProgram, "set_bnd_float");
      clSetBoundEndFloatKernel = createKernel(clProgram, "set_bnd_float_end");

      clProjectStartKernel = createKernel(clProgram, "project_start");
      clProjectEndKernel = createKernel(clProgram, "project_end");

      clAdvectFloat2Kernel = createKernel(clProgram, "advect_float2");
      clAdvectFloatKernel = createKernel(clProgram, "advect_float");
    }

    void initVBO() {
      dynarray <float>fluidPositions;
      float fluidLength = 10.0f;
      float fluidStep = fluidLength/Nborder;
      for (int i = 0; i != Nborder; i++) {
        for (int j = 0; j != Nborder; j++) {
          fluidPositions.push_back(-fluidLength/2.0f+j*fluidStep);
          fluidPositions.push_back(-fluidLength/2.0f+i*fluidStep);
          fluidPositions.push_back(0);
        }
      }
      dynarray <unsigned short>fluidIndices;
      for (int i = 0; i != N+1; i++) {
        for (int j = 0; j != N+1; j++) {
          fluidIndices.push_back(Nborder*(j+0)+(i+0));
          fluidIndices.push_back(Nborder*(j+0)+(i+1));
          fluidIndices.push_back(Nborder*(j+1)+(i+0));

          fluidIndices.push_back(Nborder*(j+0)+(i+1));
          fluidIndices.push_back(Nborder*(j+1)+(i+1));
          fluidIndices.push_back(Nborder*(j+1)+(i+0));
        }
      }
      
      glGenBuffers(1, &fluidPositionsVBO);
      glGenBuffers(1, &fluidIndicesVBO);
      
      glBindBuffer(GL_ARRAY_BUFFER, fluidPositionsVBO);
      glBufferData(GL_ARRAY_BUFFER, fluidPositions.size()*sizeof(GLfloat), (void *)fluidPositions.data(), GL_DYNAMIC_DRAW);
      glVertexPointer(3, GL_FLOAT, 0, 0);
      glEnableClientState(GL_VERTEX_ARRAY);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluidIndicesVBO);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, fluidIndices.size()*sizeof(GLushort), (void *)fluidIndices.data(), GL_DYNAMIC_DRAW);


    }

    void synch() {
      cl_int err = clFinish(clQueue);

      if (err < 0) {
        perror("Error when finishing queue");
      }
    }
    
    /*** FLUID DYNAMICS FUNCTIONS ***/
    
    void dens_step ( int N, cl_mem dens1_buffer, cl_mem dens0_buffer, cl_mem uv0_buffer, float diff, float dt )
    {
      add_source(N, dens1_buffer, dens0_buffer, dt, clAddSourceFloatKernel);
      diffuse(N, dens0_buffer, dens1_buffer, diff, dt, clLinSolveFloatKernel, clSetBoundFloatKernel, clSetBoundEndFloatKernel);
      advect(N, dens1_buffer, dens0_buffer, uv0_buffer, dt, clAdvectFloatKernel, clSetBoundFloatKernel, clSetBoundEndFloatKernel);
    }

    void vel_step ( int N, cl_mem uv1_buffer, cl_mem uv0_buffer, float visc, float dt )
    {
      add_source(N, uv1_buffer, uv0_buffer, dt, clAddSourceFloat2Kernel);
      diffuse(N, uv0_buffer, uv1_buffer, visc, dt, clLinSolveFloat2Kernel, clSetBoundFloat2Kernel, clSetBoundEndFloat2Kernel);
      project (N, uv0_buffer, dens0_buffer, dens1_buffer);
      advect(N, uv1_buffer, uv0_buffer, uv0_buffer, dt, clAdvectFloat2Kernel, clSetBoundFloat2Kernel, clSetBoundEndFloat2Kernel);
      project(N, uv1_buffer, dens0_buffer, dens1_buffer);
    }

    void add_source ( int N, cl_mem x, cl_mem s, float dt, cl_kernel clAddSourceKern)
    {
      cl_int err;

      size_t global_size[2] = {N, N};
      size_t local_size[2] = {1, 1};
      cl_int num_groups[2] = {global_size[0]/local_size[0], global_size[1]/local_size[1]};

      int size = (N+2)*(N+2);

      // Create Kernel Arguments
      err = clSetKernelArg(clAddSourceKern, 0, sizeof(cl_mem), &s);
      err |= clSetKernelArg(clAddSourceKern, 1, sizeof(cl_mem), &x);
      err |= clSetKernelArg(clAddSourceKern, 2, sizeof(cl_int), &size);
      err |= clSetKernelArg(clAddSourceKern, 3, sizeof(cl_float), &dt);
      if (err < 0) {
        perror("Could not create a kernel argument");
      }

      // Enqueue kernel

      err = clEnqueueNDRangeKernel(clQueue, clAddSourceKern, 2, NULL, global_size, local_size, 0, NULL, NULL);
      if (err < 0) {
        perror("Could not enqueue the kernel");
      }
      synch();
    }

    void set_bnd ( int N, cl_mem x, cl_kernel setBndKern, cl_kernel setBndEndKern)
    {
      cl_int err;

      size_t global_size = N;
      size_t local_size = 1;
      size_t end_size = 1;
      size_t end_size_local = 1;

      int Nborder = N + 2;

      err = clSetKernelArg(setBndKern, 0, sizeof(cl_mem), &x);
      err |= clSetKernelArg(setBndKern, 1, sizeof(cl_int), &Nborder);
      if (err < 0) {
        perror("Could not create a kernel argument for setBndKern");
        return; //exit(1);
      }

      err = clSetKernelArg(setBndEndKern, 0, sizeof(cl_mem), &x);
      err |= clSetKernelArg(setBndEndKern, 1, sizeof(int), &Nborder);
      if (err < 0) {
        perror("Could not create a kernel argument for setBndEndKern");
        return; //exit(1);
      }

      err |= clEnqueueNDRangeKernel(clQueue, setBndKern, 1, NULL, &global_size,
        &local_size, 0, NULL, NULL);
      synch();
      err |= clEnqueueNDRangeKernel(clQueue, setBndEndKern, 1, NULL, &end_size,
        &end_size_local, 0, NULL, NULL);
      if (err < 0) {
        perror("Could not enqueue the kernel for set_bnd");
        return; //exit(1);
      }
      synch();
    } 

    void lin_solve(int width, cl_mem x, cl_mem x0, float a, float c,
                   cl_kernel linSolveKern, cl_kernel setBndKern, cl_kernel setBndEndKern)
    {
      cl_int err;
      int k = 0;

      size_t global_size[2] = {width, width};
      size_t local_size[2] = {1, 1};
      cl_int num_groups[2] = {global_size[0]/local_size[0], global_size[1]/local_size[1]};
      
      int Nborder = width+2;

      // Create Kernel Arguments
      err = clSetKernelArg(linSolveKern, 0, sizeof(cl_mem), &x0);
      err |= clSetKernelArg(linSolveKern, 1, sizeof(cl_mem), &x);
      err |= clSetKernelArg(linSolveKern, 2, sizeof(cl_int), &Nborder);
      err |= clSetKernelArg(linSolveKern, 3, sizeof(cl_float), &a);
      err |= clSetKernelArg(linSolveKern, 4, sizeof(cl_float), &c);
      if (err < 0) {
        perror("Could not create a kernel argument for linSolveKern");
        return; //exit(1);
      }

      // Enqueue kernel
      for (k = 0; k != 20; k++) {
        err = clEnqueueNDRangeKernel(clQueue, linSolveKern, 2, NULL, global_size,
          local_size, 0, NULL, NULL);
        if (err < 0) {
          perror("Could not enqueue the kernel for linSolveKern");
          return; //exit(1);
        }
        synch();
        set_bnd( width, x, setBndKern, setBndEndKern );
      }
    }

    void diffuse ( int N, cl_mem x, cl_mem x0, float diff, float dt, cl_kernel linSolveKern, cl_kernel setBndKern, cl_kernel setBndEndKern)
    {
      float a=dt*diff*N*N;
      lin_solve ( N, x, x0, a, 1+4*a, linSolveKern, setBndKern, setBndEndKern);
    }
    
    void advect ( int N, cl_mem d, cl_mem d0, cl_mem uv, float dt, cl_kernel advectKern, cl_kernel setBndKern, cl_kernel setBndEndKern )
    {
      cl_int err;
      float dt0 = dt*N;
      int Nborder = N+2;

      size_t global_size[2] = {N, N};
      size_t local_size[2] = {1, 1};
      size_t end_size = 1;
      cl_int num_groups[2] = {global_size[0]/local_size[0], global_size[1]/local_size[1]};

      // Create Kernel Arguments
      err = clSetKernelArg(advectKern, 0, sizeof(cl_mem), &d0);
      err |= clSetKernelArg(advectKern, 1, sizeof(cl_mem), &d);
      err |= clSetKernelArg(advectKern, 2, sizeof(cl_mem), &uv);
      err |= clSetKernelArg(advectKern, 3, sizeof(cl_int), &Nborder);
      err |= clSetKernelArg(advectKern, 4, sizeof(cl_float), &dt0);
      if (err < 0) {
        perror("Could not create a kernel argument for advectKern");
        return; //exit(1);
      }

      err = clEnqueueNDRangeKernel(clQueue, advectKern, 2, NULL, global_size,
        local_size, 0, NULL, NULL);
      if (err < 0) {
        perror("Could not enqueue the kernel for advectKern");
        return; //exit(1);
      }
      synch();
      set_bnd ( N, d, setBndKern, setBndEndKern );
    }

    void project ( int N, cl_mem uv, cl_mem p, cl_mem div )
    {
      cl_int err;
      int k = 0;

      size_t global_size[2] = {N, N};
      size_t local_size[2] = {1, 1};
      size_t end_size = 1;
      cl_int num_groups[2] = {global_size[0]/local_size[0], global_size[1]/local_size[1]};
      
      int Nborder = N+2;
      float a = 1.0f;
      float c = 4.0f;
      
      // Create Kernel Arguments
      err = clSetKernelArg(clProjectStartKernel, 0, sizeof(cl_mem), &uv);
      err |= clSetKernelArg(clProjectStartKernel, 1, sizeof(cl_mem), &p);
      err |= clSetKernelArg(clProjectStartKernel, 2, sizeof(cl_mem), &div);
      err |= clSetKernelArg(clProjectStartKernel, 3, sizeof(cl_int), &Nborder);
      if (err < 0) {
        perror("Could not create a kernel argument for clProjectStartKernel");
        return; //exit(1);
      }

      err = clSetKernelArg(clProjectEndKernel, 0, sizeof(cl_mem), &uv);
      err |= clSetKernelArg(clProjectEndKernel, 1, sizeof(cl_mem), &p);
      err |= clSetKernelArg(clProjectEndKernel, 2, sizeof(cl_int), &Nborder);
      if (err < 0) {
        perror("Could not create a kernel argument for clProjectEndKernel");
        return; //exit(1);
      }

      err = clEnqueueNDRangeKernel(clQueue, clProjectStartKernel, 2, NULL, global_size,
          local_size, 0, NULL, NULL);
      if (err < 0) {
        perror("Could not enqueue the kernel for clProjectStartKernel");
        return; //exit(1);
      }
      synch();
      set_bnd( N, p, clSetBoundFloatKernel, clSetBoundEndFloatKernel );
      set_bnd( N, div, clSetBoundFloatKernel, clSetBoundEndFloatKernel );

      lin_solve ( N, p, div, 1.0f, 4.0f, clLinSolveFloatKernel, clSetBoundFloatKernel, clSetBoundEndFloatKernel);

      err = clEnqueueNDRangeKernel(clQueue, clProjectEndKernel, 2, NULL, global_size,
          local_size, 0, NULL, NULL);
      if (err < 0) {
        perror("Could not enqueue the kernel for clProjectEndKernel");
        return; //exit(1);
      }
      synch();
      set_bnd( N, uv, clSetBoundFloat2Kernel, clSetBoundEndFloat2Kernel );
    }

    /*** DEBUG FUNCTIONS ***/

    void print_float(float *arr, int height, int width, int num_comp) {
      printf("Printing array: [");
      int j = 0;
      int size = height*width*num_comp;
      for (int i = 0; i != size; i++) {
        if (!(i%width)) printf("\n");
        if (num_comp > 1) {
          if (j == 0) printf("(");
        }
        printf("%g, ", arr[i]);
        if (num_comp > 1) {
          if (j == num_comp-1) printf("), ");
          j = (j+1)%num_comp;
        }
      }
      printf("]\n");
    }

  public:
    // this is called when we construct the class
    engine(int argc, char **argv)
    : app(argc, argv)
    {
    }

    ~engine() {
      // Deallocate resource
      clReleaseKernel(clLinSolveFloat2Kernel);
      clReleaseMemObject(uv0_buffer);
      clReleaseMemObject(uv1_buffer);
      clReleaseMemObject(dens0_buffer);
      clReleaseMemObject(dens1_buffer);
      clReleaseCommandQueue(clQueue);
      clReleaseProgram(clProgram);
      clReleaseContext(clContext);
    }

    // this is called once OpenGL is initialized
    void app_init() {
      // set up the shaders
      fshader.init();
      cshader.init();

      N = 16;
      Nborder = N+2;
      dt = 0.1f;
      diff = 0.0f;
      visc = 0.0f;
      force = 5.0f;
      source = 100.0f;

      // Create device and context
      initOpenCL();
      
      initVBO();

      //overlay.init();
    }

    // this is called to draw the world
    void draw_world(int x, int y, int w, int h) {
      vel_step(N, uv1_buffer, uv0_buffer, visc, dt);
      dens_step(N, dens1_buffer, dens0_buffer, uv1_buffer, diff, dt);
      //print_float(uv1.data(), Nborder, Nborder, 2);
      //print_float(dens1.data(), Nborder, Nborder, 1);
      
      int vx = 0, vy = 0;
      get_viewport_size(vx, vy);

      mat4t cameraToWorld;
      cameraToWorld.translate(0, 0, 10);
      mat4t worldToCamera;
      cameraToWorld.invertQuick(worldToCamera);

      mat4t modelToWorld;
      modelToWorld.loadIdentity();

      mat4t modelToCamera = modelToWorld * worldToCamera;

      mat4t modelToProjection = mat4t::build_projection_matrix(modelToWorld, cameraToWorld, 0.1f, 1000.0f, 0.0f, 0.0f, 0.1f*vy/float(vx));

      //fshader.render(modelToProjection);
      vec4 color(1.0f, 0.0f, 0.0f, 1.0f);
      cshader.render(modelToProjection, color);
      renderFluid();
      //overlay.render(object_shader, skin_shader, vx, vy, get_frame_number());

    }

    void renderFluid() {
      glDrawElements(GL_TRIANGLES, (N+1)*(N+1)*2, GL_UNSIGNED_SHORT, 0);
    }
  };
}
