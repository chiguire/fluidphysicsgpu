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

#if defined (__APPLE__) || defined(MACOSX)
      static const char * CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
      static const char * CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif

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
    cl_kernel clLinSolveFloat2ipKernel;
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

    GLuint vertexArrayID;
    GLuint fluidPositionsVBO;
    GLuint fluidIndicesVBO;
    GLuint fluidDensity0VBO;
    GLuint fluidDensity1VBO;

    GLuint fluidVelocitiesPositionsVBO;
    GLuint fluidVelocitiesIndicesVBO;

    // Fluid data
    unsigned int N;
    unsigned int Nborder;
    float dt;
    float diff;
    float visc;
    float force;
    float source;

    float fluidLength;

    int win_x, win_y;
    int mouse_down[3];
    int omx, omy, mx, my;

    int dvel;

    float *uvArray;
    float *dArray;
    float *uvArrayPositions;

    /*** OPENCL SPECIFIC FUNCTIONS ***/
    int isExtensionSupported(const char *support_str, char *ext_string, size_t ext_buffer_size) {
      int offset = 0;
      char *next_token = NULL;

      char *token = strtok_s(ext_string, " ", &next_token);

      while (token != NULL) {
        //Check
        if (strncmp(token, support_str, ext_buffer_size) == 0) {
          return 1;
        }
        token = strtok_s(NULL, " ", &next_token);
      }
      return 0;
    }

    void initCLGLSharing() {
       //Test extension
      char ext_string[1024];
      size_t ext_size = 1024;
      cl_int err = clGetDeviceInfo(clDeviceID, CL_DEVICE_EXTENSIONS, 1024, ext_string, &ext_size);

      int supported = isExtensionSupported(CL_GL_SHARING_EXT, ext_string, ext_size);
      if (!supported) {
        printf("Not found GL Sharing Support.\n");
        return;
      }
      printf("Found GL Sharing Support.\n");
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
      cl_int err = clEnqueueWriteBuffer(clQueue, dst, CL_TRUE, 0, size*sizeof(cl_float), (void *)src, 0, NULL, NULL);
      if (err < 0) {
        printf("Could not write array.");
      }
    }

    void readArray(cl_mem src, float *dst, unsigned int size) {
      cl_int err = clEnqueueReadBuffer(clQueue, src, CL_TRUE, 0, size*sizeof(cl_float), (void *)dst, 0, NULL, NULL);
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

      cl_platform_id platform;
      cl_device_id dev;
      size_t sizet;

      // Identify a platform
      err = clGetPlatformIDs(1, &platform, NULL);
      if (err < 0) {
        perror("Could not identify a platform");
        exit(1);
      }

#if defined (__APPLE__) || defined(MACOSX)
      CGLContextObj kCGLContext = CGLGetCurrentContext();
      CGLShareGroupObj kCGLShareGroup = CGLGetShareGroup(kCGLContext);
      cl_context_properties props []= {
        CL_GL_CONTEXT_KHR, (cl_context_properties)kCGLShareGroup, 0
      };
#elif defined _WIN32 || _WIN64
      cl_context_properties props []= {
        CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
        CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
      };
      
#else
      //linux
      cl_context_properties props []= {
        CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
        CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
        CL_CONTEXT_PLATFORM, (cl_context_properties)platform,
        0
      };
#endif

      //Access a device
      clGetGLContextInfoKHR_fn leFunction = (clGetGLContextInfoKHR_fn)clGetExtensionFunctionAddress("clGetGLContextInfoKHR");
      err = leFunction(props, CL_CURRENT_DEVICE_FOR_GL_CONTEXT_KHR,
                            1*sizeof(cl_device_id), &dev, &sizet);
      if (err < 0) {
        printf("Could not access any device that could interop CL/GL");
        exit(1);
      }

      clDeviceID = dev;

      clContext = clCreateContext(props, 1, &clDeviceID, NULL, NULL, &err);
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

      int size = (Nborder)*(Nborder);
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
      //print_float(dens0.data(), Nborder, Nborder, 1);

      uv0_buffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, size * 2 * sizeof(float), uv0.data(), &err);
      //dens0_buffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE |
      //  CL_MEM_COPY_HOST_PTR, size * sizeof(float), dens0.data(), &err);
      uv1_buffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, size * 2 * sizeof(float), uv1.data(), &err);
      //dens1_buffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE |
      //  CL_MEM_COPY_HOST_PTR, size * sizeof(float), dens1.data(), &err);
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
      clLinSolveFloat2ipKernel = createKernel(clProgram, "lin_solve_float2_ip");
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
      cl_int err;

      dynarray <float>fluidPositions;
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
      
      dynarray <float>fluidDensity;
      fluidDensity.resize(Nborder*Nborder);
      for (int i = 0; i != Nborder*Nborder; i++) {
        fluidDensity[i] = 0.0f;
      }
      
      glGenVertexArrays(1, &vertexArrayID);
      glBindVertexArray(vertexArrayID);

      glGenBuffers(1, &fluidPositionsVBO);
      glGenBuffers(1, &fluidIndicesVBO);
      glGenBuffers(1, &fluidDensity0VBO);
      glGenBuffers(1, &fluidDensity1VBO);
      glGenBuffers(1, &fluidVelocitiesPositionsVBO);
      glGenBuffers(1, &fluidVelocitiesIndicesVBO);
      
      glBindBuffer(GL_ARRAY_BUFFER, fluidPositionsVBO);
      glBufferData(GL_ARRAY_BUFFER, fluidPositions.size()*sizeof(GLfloat), (void *)fluidPositions.data(), GL_DYNAMIC_DRAW);
      glVertexPointer(3, GL_FLOAT, 0, 0);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluidIndicesVBO);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, fluidIndices.size()*sizeof(GLushort), (void *)fluidIndices.data(), GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ARRAY_BUFFER, fluidDensity0VBO);
      glBufferData(GL_ARRAY_BUFFER, fluidDensity.size()*sizeof(GLfloat), (void *)fluidDensity.data(), GL_DYNAMIC_DRAW);
      glVertexPointer(1, GL_FLOAT, 0, 0);
      dens0_buffer = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, fluidDensity0VBO, &err);
      if (err < 0) {
        perror("Error creating CL buffer from GL.");
      }

      glBindBuffer(GL_ARRAY_BUFFER, fluidDensity1VBO);
      glBufferData(GL_ARRAY_BUFFER, fluidDensity.size()*sizeof(GLfloat), (void *)fluidDensity.data(), GL_DYNAMIC_DRAW);
      glVertexPointer(1, GL_FLOAT, 0, 0);
      dens1_buffer = clCreateFromGLBuffer(clContext, CL_MEM_READ_WRITE, fluidDensity1VBO, &err);
      if (err < 0) {
        perror("Error creating CL buffer from GL.");
      }

      for (int j = 0; j != Nborder; j++) {
        for (int i = 0; i != Nborder; i++) {
          uvArrayPositions[(j*Nborder+i)*6+0] = -fluidLength/2.0f+i*fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+1] = -fluidLength/2.0f+j*fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+2] = 0.0f;

          uvArrayPositions[(j*Nborder+i)*6+3] = -fluidLength/2.0f+i*fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+4] = -fluidLength/2.0f+j*fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+5] = 0.0f;
        }
      }

      dynarray<unsigned short>fluidVelocitiesIndices;
      for (unsigned short i = 0; i != Nborder*Nborder; i++) {
        fluidVelocitiesIndices.push_back(i*2);
        fluidVelocitiesIndices.push_back(i*2+1);
      }

      glBindBuffer(GL_ARRAY_BUFFER, fluidVelocitiesPositionsVBO);
      glBufferData(GL_ARRAY_BUFFER, Nborder*Nborder*6*sizeof(GLfloat), (void *)uvArrayPositions, GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluidVelocitiesIndicesVBO);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, fluidVelocitiesIndices.size()*sizeof(GLushort), (void *)fluidVelocitiesIndices.data(), GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ARRAY_BUFFER, 0);
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    }

    void synch() {
      cl_int err = clFlush(clQueue);

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
      project(N, uv0_buffer, uv1_buffer);
      advect(N, uv1_buffer, uv0_buffer, uv0_buffer, dt, clAdvectFloat2Kernel, clSetBoundFloat2Kernel, clSetBoundEndFloat2Kernel);
      project(N, uv1_buffer, uv0_buffer);
    }

    void add_source ( int N, cl_mem x, cl_mem s, float dt, cl_kernel clAddSourceKern)
    {
      cl_int err;

      size_t global_size[2] = {N, N};
      size_t local_size[2] = {1, 1};
      cl_int num_groups[2] = {global_size[0]/local_size[0], global_size[1]/local_size[1]};

      int size = (N+2);

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
      err |= clEnqueueNDRangeKernel(clQueue, setBndEndKern, 1, NULL, &end_size,
        &end_size_local, 0, NULL, NULL);
      if (err < 0) {
        perror("Could not enqueue the kernel for set_bnd");
        return; //exit(1);
      }
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
      for (k = 0; k != 50; k++) {
        err = clEnqueueNDRangeKernel(clQueue, linSolveKern, 2, NULL, global_size,
          local_size, 0, NULL, NULL);
        if (err < 0) {
          perror("Could not enqueue the kernel for linSolveKern");
          return; //exit(1);
        }
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
      set_bnd ( N, d, setBndKern, setBndEndKern );
    }

    void project ( int N, cl_mem uv1, cl_mem uv0 )
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
      err = clSetKernelArg(clProjectStartKernel, 0, sizeof(cl_mem), &uv1);
      err |= clSetKernelArg(clProjectStartKernel, 1, sizeof(cl_mem), &uv0);
      err |= clSetKernelArg(clProjectStartKernel, 2, sizeof(cl_int), &Nborder);
      if (err < 0) {
        perror("Could not create a kernel argument for clProjectStartKernel");
        return; //exit(1);
      }

      err = clSetKernelArg(clProjectEndKernel, 0, sizeof(cl_mem), &uv1);
      err |= clSetKernelArg(clProjectEndKernel, 1, sizeof(cl_mem), &uv0);
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
      set_bnd( N, uv0, clSetBoundFloat2Kernel, clSetBoundEndFloat2Kernel );

      lin_solve ( N, uv0, uv0, 1.0f, 4.0f, clLinSolveFloat2ipKernel, clSetBoundFloat2Kernel, clSetBoundEndFloat2Kernel);

      err = clEnqueueNDRangeKernel(clQueue, clProjectEndKernel, 2, NULL, global_size,
          local_size, 0, NULL, NULL);
      if (err < 0) {
        perror("Could not enqueue the kernel for clProjectEndKernel");
        return; //exit(1);
      }
      set_bnd( N, uv1, clSetBoundFloat2Kernel, clSetBoundEndFloat2Kernel );
    }

    /*** UI FUNCTIONS ***/
    void get_from_UI ( float * d, float * uv )
    {
      int i, j, size = (N+2)*(N+2);

      memset(uv, 0, size*2*sizeof(float));
      memset(d, 0, size*sizeof(float));

      if ( !mouse_down[0] && !mouse_down[2] ) return;

      i = (int)((       mx /(float)win_x)*N+1);
      j = (int)(((win_y-my)/(float)win_y)*N+1);
      
      if ( i<1 || i>N || j<1 || j>N ) return;

      if ( mouse_down[0] ) {
        uv[j*2*Nborder+i*2]   = force * (mx-omx);
        uv[j*2*Nborder+i*2+1] = force * (omy-my);
        printf("Force: (%g, %g)\n", uv[j*2*Nborder+i*2], uv[j*2*Nborder+i*2+1]);
      }

      if ( mouse_down[2] ) {
        d[j*Nborder+i] = source;
        d[(j-1)*Nborder+(i+0)] = source;
        d[(j+1)*Nborder+(i+0)] = source;
        d[(j+0)*Nborder+(i-1)] = source;
        d[(j+0)*Nborder+(i+1)] = source;
      }

      omx = mx;
      omy = my;

      return;
    }

    /*** DEBUG FUNCTIONS ***/

    void print_float(float *arr, int height, int width, int num_comp) {
      printf("Printing array: [");
      int j = 0;
      int size = height*width*num_comp;
      for (int i = 0; i != size; i++) {
        if (!(i%(width*num_comp))) printf("\n");
        if (num_comp > 1) {
          if (j == 0) printf("(");
        }
        printf("%.2f, ", arr[i]);
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
      free(uvArray);
      free(dArray);
      free(uvArrayPositions);
    }

    // this is called once OpenGL is initialized
    void app_init() {
      // set up the shaders
      fshader.init();
      cshader.init();

      N = 64; 
      Nborder = N+2;
      dt = 0.1f;
      diff = 0.0f;
      visc = 0.0f;
      force = 10.0f;
      source = 100.0f;

      fluidLength = 19.0f;

      dvel = 0;

      uvArray = (float *)malloc(Nborder*Nborder*sizeof(float)*2);
      dArray = (float *)malloc(Nborder*Nborder*sizeof(float));
      uvArrayPositions = (float *)malloc(Nborder*Nborder*3*2*sizeof(float));

      // Create device and context
      initOpenCL();
      initVBO();
      initCLGLSharing();

      //overlay.init();
    }

    void readMouse() {
      omx = mx;
      omy = my;
      get_mouse_pos(mx, my);
      
      mouse_down[0] = is_key_down(key_lmb);
      mouse_down[1] = is_key_down(key_mmb);
      mouse_down[2] = is_key_down(key_rmb);

      if (is_key_down('Z')) {
        dvel = dvel? 0: 1;
        printf("Changing dvel to %d\n", dvel);
      }
    }

    // this is called to draw the world
    void draw_world(int x, int y, int w, int h) {
      int vx = 0, vy = 0;
      get_viewport_size(vx, vy);
      win_x = vx;
      win_y = vy;

      glClearColor(0.1f, 0.1f, 0.2f, 1.0f);
      glClear(GL_COLOR_BUFFER_BIT);

      readMouse();
      calculateFluid();
      
      mat4t modelToProjection = setProjection(vx, vy);

      fshader.render(modelToProjection);
      renderFluid();
      if (dvel) {
        vec4 color(1.0f, 0.0f, 0.0f, 1.0f);
        cshader.render(modelToProjection, color);
        renderVelocities();
      }
      //overlay.render(object_shader, skin_shader, vx, vy, get_frame_number());
    }

   mat4t setProjection(int vx, int vy) {
      mat4t cameraToWorld;
      cameraToWorld.translate(0, 0, 10);
      mat4t worldToCamera;
      cameraToWorld.invertQuick(worldToCamera);

      mat4t modelToWorld;
      modelToWorld.loadIdentity();

      mat4t modelToCamera = modelToWorld * worldToCamera;
      return mat4t::build_projection_matrix(modelToWorld, cameraToWorld, 0.1f, 1000.0f, 0.0f, 0.0f, 0.1f*vy/float(vx));
    } 

    void calculateFluid() {
      cl_int err;

      err = clEnqueueAcquireGLObjects(clQueue, 1, &dens0_buffer, NULL, NULL, NULL);
      err = clEnqueueAcquireGLObjects(clQueue, 1, &dens1_buffer, NULL, NULL, NULL);
      if (err < 0) {
        perror("Error acquiring GL objects.");
      }
      get_from_UI ( dArray, uvArray );
      writeArray(dens0_buffer, dArray, Nborder*Nborder);
      writeArray(uv0_buffer, uvArray, Nborder*Nborder*2);

      vel_step(N, uv1_buffer, uv0_buffer, visc, dt);
      dens_step(N, dens1_buffer, dens0_buffer, uv1_buffer, diff, dt);

      /*readArray(dens1_buffer, dArray, Nborder*Nborder);
      readArray(uv1_buffer, uvArray, Nborder*Nborder*2);

      glBindBuffer(GL_ARRAY_BUFFER, fluidDensity0VBO);
      glBufferSubData(GL_ARRAY_BUFFER, 0, Nborder*Nborder*sizeof(GLfloat), dArray);*/

      err = clEnqueueReleaseGLObjects(clQueue, 1, &dens0_buffer, NULL, NULL, NULL);
      err = clEnqueueReleaseGLObjects(clQueue, 1, &dens1_buffer, NULL, NULL, NULL);
      if (err < 0) {
        perror("Error releasing GL objects.");
      }
      synch();
    }

    void renderFluid() {
      glEnableVertexAttribArray(attribute_pos);
      glBindBuffer(GL_ARRAY_BUFFER, fluidPositionsVBO);
      glVertexAttribPointer(attribute_pos, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
      
      glEnableVertexAttribArray(attribute_uv);
      glBindBuffer(GL_ARRAY_BUFFER, fluidDensity1VBO);
      glVertexAttribPointer(attribute_uv, 1, GL_FLOAT, GL_FALSE, 0, (void *)0);
      
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluidIndicesVBO);
      glDrawElements(GL_TRIANGLES, (N+1)*(N+1)*2*3, GL_UNSIGNED_SHORT, 0);
      
      glFlush();

      glDisableVertexAttribArray(attribute_pos);
      glDisableVertexAttribArray(attribute_uv);
    }

    void renderVelocities() {
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluidVelocitiesIndicesVBO);
      glBindBuffer(GL_ARRAY_BUFFER, fluidVelocitiesPositionsVBO);

      readArray(uv1_buffer, uvArray, Nborder*Nborder*2);
      float fluidStep = fluidLength/Nborder;

      for (int j = 0; j != Nborder; j++) {
        for (int i = 0; i != Nborder; i++) {
          uvArrayPositions[(j*Nborder+i)*6+0] = -fluidLength/2.0f+i*fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+1] = -fluidLength/2.0f+j*fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+2] = 0.0f;

          uvArrayPositions[(j*Nborder+i)*6+3] = -fluidLength/2.0f+i*fluidStep + uvArray[(j*Nborder+i)*2]/fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+4] = -fluidLength/2.0f+j*fluidStep + uvArray[(j*Nborder+i)*2+1]/fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+5] = 0.0f;
        }
      }
      glBufferSubData(GL_ARRAY_BUFFER, 0, Nborder*Nborder*6*sizeof(GLfloat), uvArrayPositions);
      glLineWidth(1.5f);
      glEnableVertexAttribArray(attribute_pos);
      glVertexAttribPointer(attribute_pos, 3, GL_FLOAT, GL_FALSE, 0, 0);
      glDrawElements(GL_LINES, Nborder*Nborder*2, GL_UNSIGNED_SHORT, 0);
      glFlush();
      glDisableVertexAttribArray(attribute_pos);
    }
  };
}
