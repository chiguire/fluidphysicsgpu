////////////////////////////////////////////////////////////////////////////////
//
// (C) Andy Thomason 2012-2014
//
// Modular Framework for OpenGLES2 rendering on multiple platforms.
//

#define IX(i,j) ((i)+(N+2)*(j))
#define SWAP(x0,x) {float * tmp=x0;x0=x;x=tmp;}
#define FOR_EACH_CELL for ( i=1 ; i<=N ; i++ ) { for ( j=1 ; j<=N ; j++ ) {
#define END_FOR }}

namespace octet {
  /// Scene containing a box with octet.
  class fluidshader : public app {
    fluid_shader fshader;
    color_shader cshader;

    int N;
    int Nborder;
    float dt, diff, visc;
    float force, source;
    int dvel;
    int currentAngle;

    float * u, * v, * u_prev, * v_prev;
    float * dens, * dens_prev;

    float *uvArrayPositions;

    int win_x, win_y;
    int mouse_down[3];
    int omx, omy, mx, my;

    GLuint vertexArrayID;
    GLuint fluidPositionsVBO;
    GLuint fluidIndicesVBO;
    GLuint fluidDensity0VBO;
    GLuint fluidDensity1VBO;

    GLuint fluidVelocitiesPositionsVBO;
    GLuint fluidVelocitiesIndicesVBO;

    void initVBO() {
      dynarray <float>fluidPositions;
      float fluidLength = 18.0f;
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

      glBindBuffer(GL_ARRAY_BUFFER, fluidDensity0VBO);
      glBufferData(GL_ARRAY_BUFFER, fluidDensity.size()*sizeof(GLfloat), (void *)fluidDensity.data(), GL_DYNAMIC_DRAW);

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluidIndicesVBO);
      glBufferData(GL_ELEMENT_ARRAY_BUFFER, fluidIndices.size()*sizeof(GLushort), (void *)fluidIndices.data(), GL_DYNAMIC_DRAW);

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

    void free_data ( void )
    {
      if ( u ) free ( u );
      if ( v ) free ( v );
      if ( u_prev ) free ( u_prev );
      if ( v_prev ) free ( v_prev );
      if ( dens ) free ( dens );
      if ( dens_prev ) free ( dens_prev );
      if ( uvArrayPositions ) free ( uvArrayPositions );
    }

    void clear_data ( void )
    {
      int i, size=(N+2)*(N+2);

      for ( i=0 ; i<size ; i++ ) {
        u[i] = v[i] = u_prev[i] = v_prev[i] = dens[i] = dens_prev[i] = 0.0f;
      }
    }

    int allocate_data ( void )
    {
      int size = (N+2)*(N+2);

      u			= (float *) malloc ( size*sizeof(float) );
      v			= (float *) malloc ( size*sizeof(float) );
      u_prev		= (float *) malloc ( size*sizeof(float) );
      v_prev		= (float *) malloc ( size*sizeof(float) );
      dens		= (float *) malloc ( size*sizeof(float) );	
      dens_prev	= (float *) malloc ( size*sizeof(float) );
      uvArrayPositions = (float *)malloc(size*3*2*sizeof(float));

      if ( !u || !v || !u_prev || !v_prev || !dens || !dens_prev ) {
        fprintf ( stderr, "cannot allocate data\n" );
        return ( 0 );
      }

      return ( 1 );
    }

    void get_from_UI ( float * d, float * u, float * v )
    {
      int i, j, size = (N+2)*(N+2);

      for ( i=0 ; i<size ; i++ ) {
        u[i] = v[i] = d[i] = 0.0f;
      }

      if ( !mouse_down[0] && !mouse_down[2] ) return;

      i = (int)((       mx /(float)win_x)*N+1);
      j = (int)(((win_y-my)/(float)win_y)*N+1);
      
      if ( i<1 || i>N || j<1 || j>N ) return;

      if ( mouse_down[0] ) {
        printf("Force: (%d, %d)\n", (mx-omx), (omy-my));
        u[IX(i,j)] = force * (mx-omx);
        v[IX(i,j)] = force * (omy-my);
      }

      if ( mouse_down[2] ) {
        d[IX(i,j)] = source;
      }

      omx = mx;
      omy = my;

      return;
    }

    void add_source ( int N, float * x, float * s, float dt )
    {
      int i, size=(N+2)*(N+2);
      for ( i=0 ; i<size ; i++ ) x[i] += dt*s[i];
    }

    void set_bnd ( int N, int b, float * x )
    {
      int i;

      for ( i=1 ; i<=N ; i++ ) {
        x[IX(0  ,i)] = b==1 ? -x[IX(1,i)] : x[IX(1,i)];
        x[IX(N+1,i)] = b==1 ? -x[IX(N,i)] : x[IX(N,i)];
        x[IX(i,0  )] = b==2 ? -x[IX(i,1)] : x[IX(i,1)];
        x[IX(i,N+1)] = b==2 ? -x[IX(i,N)] : x[IX(i,N)];
      }
      x[IX(0  ,0  )] = 0.5f*(x[IX(1,0  )]+x[IX(0  ,1)]);
      x[IX(0  ,N+1)] = 0.5f*(x[IX(1,N+1)]+x[IX(0  ,N)]);
      x[IX(N+1,0  )] = 0.5f*(x[IX(N,0  )]+x[IX(N+1,1)]);
      x[IX(N+1,N+1)] = 0.5f*(x[IX(N,N+1)]+x[IX(N+1,N)]);
    }

    void lin_solve ( int N, int b, float * x, float * x0, float a, float c )
    {
      int i, j, k;

      for ( k=0 ; k<20 ; k++ ) {
        FOR_EACH_CELL
          x[IX(i,j)] = (x0[IX(i,j)] + a*(x[IX(i-1,j)]+x[IX(i+1,j)]+x[IX(i,j-1)]+x[IX(i,j+1)]))/c;
        END_FOR
          set_bnd ( N, b, x );
      }
    }

    void diffuse ( int N, int b, float * x, float * x0, float diff, float dt )
    {
      float a=dt*diff*N*N;
      lin_solve ( N, b, x, x0, a, 1+4*a );
    }

    void advect ( int N, int b, float * d, float * d0, float * u, float * v, float dt )
    {
      int i, j, i0, j0, i1, j1;
      float x, y, s0, t0, s1, t1, dt0;

      dt0 = dt*N;
      FOR_EACH_CELL
        x = i-dt0*u[IX(i,j)]; y = j-dt0*v[IX(i,j)];
      if (x<0.5f) x=0.5f; if (x>N+0.5f) x=N+0.5f; i0=(int)x; i1=i0+1;
      if (y<0.5f) y=0.5f; if (y>N+0.5f) y=N+0.5f; j0=(int)y; j1=j0+1;
      s1 = x-i0; s0 = 1-s1; t1 = y-j0; t0 = 1-t1;
      d[IX(i,j)] = s0*(t0*d0[IX(i0,j0)]+t1*d0[IX(i0,j1)])+
        s1*(t0*d0[IX(i1,j0)]+t1*d0[IX(i1,j1)]);
      END_FOR
        set_bnd ( N, b, d );
    }

    void project ( int N, float * u, float * v, float * p, float * div )
    {
      int i, j;

      FOR_EACH_CELL
        div[IX(i,j)] = -0.5f*(u[IX(i+1,j)]-u[IX(i-1,j)]+v[IX(i,j+1)]-v[IX(i,j-1)])/N;
      p[IX(i,j)] = 0;
      END_FOR	
        set_bnd ( N, 0, div ); set_bnd ( N, 0, p );

      lin_solve ( N, 0, p, div, 1, 4 );

      FOR_EACH_CELL
        u[IX(i,j)] -= 0.5f*N*(p[IX(i+1,j)]-p[IX(i-1,j)]);
      v[IX(i,j)] -= 0.5f*N*(p[IX(i,j+1)]-p[IX(i,j-1)]);
      END_FOR
        set_bnd ( N, 1, u ); set_bnd ( N, 2, v );
    }

    void dens_step ( int N, float * x, float * x0, float * u, float * v, float diff, float dt )
    {
      add_source ( N, x, x0, dt );
      SWAP ( x0, x ); diffuse ( N, 0, x, x0, diff, dt );
      SWAP ( x0, x ); advect ( N, 0, x, x0, u, v, dt );
    }

    void vel_step ( int N, float * u, float * v, float * u0, float * v0, float visc, float dt )
    {
      add_source ( N, u, u0, dt ); add_source ( N, v, v0, dt );
      diffuse ( N, 1, u0, u, visc, dt );
      diffuse ( N, 2, v0, v, visc, dt );
      project ( N, u0, v0, u, v );
      advect ( N, 1, u, u0, u0, v0, dt ); advect ( N, 2, v, v0, u0, v0, dt );
      project ( N, u, v, u0, v0 );
    }


  public:
    /// this is called when we construct the class before everything is initialised.
    fluidshader(int argc, char **argv) : app(argc, argv) {
    }

    /// this is called once OpenGL is initialized
    void app_init() {
      fshader.init();
      cshader.init();

      N = 64;
      Nborder = N+2;
      dt = 0.1f;
      diff = 0.0f;
      visc = 0.0f;
      force = 5.0f;
      source = 100.0f;

      dvel = 0;
      currentAngle = 0;
      if ( !allocate_data () ) exit ( 1 );
      clear_data ();

      initVBO();
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

    /// this is called to draw the world
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
        vec4 color(1.0f, 1.0f, 0.0f, 1.0f);
        cshader.render(modelToProjection, color);
        renderVelocities();
      }
    }

    mat4t setProjection(int vx, int vy) {
      mat4t cameraToWorld;
      cameraToWorld.translate(0, 0, 10);
      mat4t worldToCamera;
      cameraToWorld.invertQuick(worldToCamera);

      mat4t modelToWorld;
      modelToWorld.loadIdentity();

      mat4t modelToCamera = modelToWorld * worldToCamera;
      return mat4t::build_projection_matrix(modelToWorld, cameraToWorld);
    } 

    void calculateFluid() {
      get_from_UI ( dens_prev, u_prev, v_prev );
      vel_step ( N, u, v, u_prev, v_prev, visc, dt );
      dens_step ( N, dens, dens_prev, u, v, diff, dt );

      glBindBuffer(GL_ARRAY_BUFFER, fluidDensity0VBO);
      glBufferSubData(GL_ARRAY_BUFFER, 0, Nborder*Nborder*sizeof(GLfloat), dens);
    }

    void renderFluid() {
      glEnableVertexAttribArray(attribute_pos);
      glBindBuffer(GL_ARRAY_BUFFER, fluidPositionsVBO);
      glVertexAttribPointer(attribute_pos, 3, GL_FLOAT, GL_FALSE, 0, (void *)0);
      
      glEnableVertexAttribArray(attribute_uv);
      glBindBuffer(GL_ARRAY_BUFFER, fluidDensity0VBO);
      glVertexAttribPointer(attribute_uv, 1, GL_FLOAT, GL_FALSE, 0, (void *)0);
      
      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluidIndicesVBO);
      glDrawElements(GL_TRIANGLES, (N+1)*(N+1)*2*3, GL_UNSIGNED_SHORT, 0);
      
      glFlush();

      glDisableVertexAttribArray(attribute_pos);
      glDisableVertexAttribArray(attribute_uv);
    }

    void renderVelocities() {
      float fluidLength = 18.0f;
      float fluidStep = fluidLength/Nborder;

      for (int j = 0; j != Nborder; j++) {
        for (int i = 0; i != Nborder; i++) {
          uvArrayPositions[(j*Nborder+i)*6+0] = -fluidLength/2.0f+i*fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+1] = -fluidLength/2.0f+j*fluidStep;
          uvArrayPositions[(j*Nborder+i)*6+2] = 0.0f;

          uvArrayPositions[(j*Nborder+i)*6+3] = -fluidLength/2.0f+i*fluidStep + u[IX(i, j)];
          uvArrayPositions[(j*Nborder+i)*6+4] = -fluidLength/2.0f+j*fluidStep + v[IX(i, j)];
          uvArrayPositions[(j*Nborder+i)*6+5] = 0.0f;
        }
      }

      glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, fluidVelocitiesIndicesVBO);
      glBindBuffer(GL_ARRAY_BUFFER, fluidVelocitiesPositionsVBO);
      glBufferSubData(GL_ARRAY_BUFFER, 0, Nborder*Nborder*6*sizeof(GLfloat), uvArrayPositions);
      glLineWidth(1.5f);
      glEnableVertexAttribArray(attribute_pos);
      glVertexAttribPointer(attribute_pos, 3, GL_FLOAT, GL_FALSE, 0, 0);
      glDrawElements(GL_LINES, Nborder*Nborder*2, GL_UNSIGNED_SHORT, 0);
      glDisableVertexAttribArray(attribute_pos);
    }
  };
}
