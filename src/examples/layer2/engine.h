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
    bump_shader object_shader;

    // helper for drawing text
    //text_overlay overlay;

    // OpenCL stuff
    cl_device_id clDeviceID;
    cl_context clContext;
    cl_program clProgram;
    cl_kernel clKernel;
    cl_command_queue clQueue;

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
      }
      if (err < 0) {
        printf("Could not access any device");
        exit(1);
      }

      return dev;
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
      err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
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

  public:
    // this is called when we construct the class
    engine(int argc, char **argv) : app(argc, argv) {
    }

    // this is called once OpenGL is initialized
    void app_init() {
      // set up the shaders
      object_shader.init(false);

      cl_int err;
      size_t local_size, global_size;

      float data[64];
      float sum[2], total, actual_sum;
      cl_mem input_buffer, sum_buffer;
      cl_int num_groups;

      for (int i = 0; i < 64; i ++) {
        data[i] = 1.0f*i;
      }
      
      // Create device and context
      clDeviceID = createDevice();
      clContext = clCreateContext(NULL, 1, &clDeviceID, NULL, NULL, &err);
      if (err < 0) {
        perror("Could not create a context");
        return; //exit(1);
      }

      // Build program
      clProgram = buildProgram(clContext, clDeviceID, "assets/add_numbers.cl");

      // Create data buffer
      global_size = 8;
      local_size = 4;
      num_groups = global_size/local_size;
      input_buffer = clCreateBuffer(clContext, CL_MEM_READ_ONLY |
        CL_MEM_COPY_HOST_PTR, 64 * sizeof(float), data, &err);
      sum_buffer = clCreateBuffer(clContext, CL_MEM_READ_WRITE |
        CL_MEM_COPY_HOST_PTR, num_groups * sizeof(float), sum, &err);
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
      clKernel = clCreateKernel(clProgram, "add_numbers", &err);
      if (err < 0) {
        perror("Could not create a kernel");
        return; //exit(1);
      }

      // Create Kernel Arguments
      err = clSetKernelArg(clKernel, 0, sizeof(cl_mem), &input_buffer);
      err |= clSetKernelArg(clKernel, 1, local_size * sizeof(float), NULL);
      err |= clSetKernelArg(clKernel, 2, sizeof(cl_mem), &sum_buffer);
      if (err < 0) {
        perror("Could not create a kernel argument");
        return; //exit(1);
      }

      // Enqueue kernel
      err = clEnqueueNDRangeKernel(clQueue, clKernel, 1, NULL, &global_size,
        &local_size, 0, NULL, NULL);
      if (err < 0) {
        perror("Could not enqueue the kernel");
        return; //exit(1);
      }

      // Read Kernel's output
      err = clEnqueueReadBuffer(clQueue, sum_buffer, CL_TRUE, 0,
        sizeof(sum), sum, 0, NULL, NULL);
      if (err < 0) {
        perror("Could not read the buffer");
        return; //exit(1);
      }

      // Check result
      total = 0.0f;
      for (int j = 0; j < num_groups; j++) {
        total += sum[j];
      }
      actual_sum = 1.0f * 64/2*(64-1);
      printf("Computed sum = %.2f.\n", total);
      if (fabs(total - actual_sum) > 0.01*fabs(actual_sum))
        printf("Check failed.\n");
      else
        printf("Check passed.\n");

      // Deallocate resource
      clReleaseKernel(clKernel);
      clReleaseMemObject(sum_buffer);
      clReleaseMemObject(input_buffer);
      clReleaseCommandQueue(clQueue);
      clReleaseProgram(clProgram);
      clReleaseContext(clContext);

      //overlay.init();
    }

    // this is called to draw the world
    void draw_world(int x, int y, int w, int h) {

      //int vx = 0, vy = 0;
      //get_viewport_size(vx, vy);
      //overlay.render(object_shader, skin_shader, vx, vy, get_frame_number());
    }
  };
}
