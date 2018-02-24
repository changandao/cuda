// ###
// ###
// ### Practical Course: GPU Programming in Computer Vision
// ###
// ###
// ### Technical University Munich, Computer Vision Group
// ### Summer Semester 2017, September 11 - October 9
// ###

#include "helper.h"
#include <iostream>
#include "math.h"
using namespace std;

// uncomment to use the camera
//#define CAMERA


// Exercise 4
__global__ void getGradient( float *gradientx, float *gradienty, float *img, float w, float h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    for(int c = 0; c < nc; c++)
    {
        size_t ind = x + (size_t)y*w + (size_t)w*h*c;
        size_t indxp = x+1 + (size_t)y*w + (size_t)w*h*c;
        size_t indyp = x + (size_t)(y+1)*w + (size_t)w*h*c;
        float resx = 0;
        float resy = 0;
        int n = w*h*nc;
        if(ind+1 < n)
        {
            //gradient x
            float xplus1 = img[indxp];
            float x0 = img[ind];
            resx = xplus1 - x0;
            //gradient y
            float yplus1 = img[indyp];
            float y0 = img[ind];
            resy = yplus1 - y0;
        }
        if(ind<n){
            gradientx[ind] = resx;
            gradienty[ind] = resy;
        }
    }

}

__global__ void getDivergence(float *divergence, float *img, float w, float h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    for(int c = 0; c < nc; c++)
    {
        size_t ind = x + (size_t)y*w + (size_t)w*h*c;
        size_t indxp = x+1 + (size_t)y*w + (size_t)w*h*c;
        size_t indyp = x + (size_t)(y+1)*w + (size_t)w*h*c;
        float resx = 0;
        float resy = 0;
        int n = w*h*nc;
        if(ind+1 < n)
        {
            //gradient x
            float xplus1 = img[indxp];
            float x0 = img[ind];
            resx = xplus1 - x0;
            //gradient y
            float yplus1 = img[indyp];
            float y0 = img[ind];
            resy = yplus1 - y0;
        }
        if(ind<n){
            divergence[ind] = resx + resy;
        }
    }
}

__global__ void getLaplacian( float *Laplacian, float *img, float w, float h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    for(int c = 0; c < nc; c++)
    {
        size_t ind = x + (size_t)y*w + (size_t)w*h*c;
        size_t indxp = x+1 + (size_t)y*w + (size_t)w*h*c;
        size_t indyp = x + (size_t)(y+1)*w + (size_t)w*h*c;

        float resx = 0;
        float resy = 0;
        int n = w*h*nc;
        if(ind+1 < n)
        {
            //gradient x
            float xplus1 = img[indxp];
            float x0 = img[ind];
            resx = xplus1 - x0;
            //gradient y
            float yplus1 = img[indyp];
            float y0 = img[ind];
            resy = yplus1 - y0;
        }

        if(ind<n){
            Laplacian[ind] = sqrt(resx * resx + resy * resy);
        }
    }

}



int main(int argc, char **argv)
{
    // Before the GPU can process your kernels, a so called "CUDA context" must be initialized
    // This happens on the very first call to a CUDA function, and takes some time (around half a second)
    // We will do it right here, so that the run time measurements are accurate
    cudaDeviceSynchronize();  CUDA_CHECK;




    // Reading command line parameters:
    // getParam("param", var, argc, argv) looks whether "-param xyz" is specified, and if so stores the value "xyz" in "var"
    // If "-param" is not specified, the value of "var" remains unchanged
    //
    // return value: getParam("param", ...) returns true if "-param" is specified, and false otherwise

#ifdef CAMERA
#else
    // input image
    string image = "";
    bool ret = getParam("i", image, argc, argv);
    if (!ret) cerr << "ERROR: no image specified" << endl;
    if (argc <= 1) { cout << "Usage: " << argv[0] << " -i <image> [-repeats <repeats>] [-gray]" << endl; return 1; }
#endif

    // number of computation repetitions to get a better run time measurement
    int repeats = 1;
    getParam("repeats", repeats, argc, argv);
    cout << "repeats: " << repeats << endl;

    // load the input image as grayscale if "-gray" is specifed
    bool gray = false;
    getParam("gray", gray, argc, argv);
    cout << "gray: " << gray << endl;

    // ### Define your own parameters here as needed

    // Init camera / Load input image
#ifdef CAMERA

    // Init camera
    cv::VideoCapture camera(0);
    if(!camera.isOpened()) { cerr << "ERROR: Could not open camera" << endl; return 1; }
    int camW = 640;
    int camH = 480;
    camera.set(CV_CAP_PROP_FRAME_WIDTH,camW);
    camera.set(CV_CAP_PROP_FRAME_HEIGHT,camH);
    // read in first frame to get the dimensions
    cv::Mat mIn;
    camera >> mIn;

#else

    // Load the input image using opencv (load as grayscale if "gray==true", otherwise as is (may be color or grayscale))
    cv::Mat mIn = cv::imread(image.c_str(), (gray? CV_LOAD_IMAGE_GRAYSCALE : -1));
    // check
    if (mIn.data == NULL) { cerr << "ERROR: Could not load image " << image << endl; return 1; }

#endif

    // convert to float representation (opencv loads image values as single bytes by default)
    mIn.convertTo(mIn,CV_32F);
    // convert range of each channel to [0,1] (opencv default is [0,255])
    mIn /= 255.f;
    // get image dimensions
    int w = mIn.cols;         // width
    int h = mIn.rows;         // height
    int nc = mIn.channels();  // number of channels
    cout << "image: " << w << " x " << h << endl;




    // Set the output image format
    // ###
    // ###
    // ### TODO: Change the output image format as needed
    // ###
    // ###
    cv::Mat mOut(h,w,mIn.type());  // mOut will have the same number of channels as the input image, nc layers
    //cv::Mat mOut(h,w,CV_32FC3);    // mOut will be a color image, 3 layers
    //cv::Mat mOut(h,w,CV_32FC1);    // mOut will be a grayscale image, 1 layer
    // ### Define your own output images here as needed
    if (nc == 1)
    {
        cv::Mat mOut(h,w,CV_32FC1);
    }
    else{cv::Mat mOut(h,w,CV_32FC3);}


    // Allocate arrays
    // input/output image width: w
    // input/output image height: h
    // input image number of channels: nc
    // output image number of channels: mOut.channels(), as defined above (nc, 3, or 1)

    // allocate raw input image array
    float *imgIn = new float[(size_t)w*h*nc];

    // allocate raw output array (the computation result will be stored in this array, then later converted to mOut for displaying)
    float *imgOut = new float[(size_t)w*h*mOut.channels()];




    // For camera mode: Make a loop to read in camera frames
#ifdef CAMERA
    // Read a camera image frame every 30 milliseconds:
    // cv::waitKey(30) waits 30 milliseconds for a keyboard input,
    // returns a value <0 if no key is pressed during this time, returns immediately with a value >=0 if a key is pressed
    while (cv::waitKey(30) < 0)
    {
        // Get camera image
        camera >> mIn;
        // convert to float representation (opencv loads image values as single bytes by default)
        mIn.convertTo(mIn,CV_32F);
        // convert range of each channel to [0,1] (opencv default is [0,255])
        mIn /= 255.f;
#endif



    // Init raw input image array
    // opencv images are interleaved: rgb rgb rgb...  (actually bgr bgr bgr...)
    // But for CUDA it's better to work with layered images: rrr... ggg... bbb...
    // So we will convert as necessary, using interleaved "cv::Mat" for loading/saving/displaying, and layered "float*" for CUDA computations
    convert_mat_to_layered (imgIn, mIn);


    //Exercise 4
    int version = 1;
    Timer timer; timer.start();
    int len = w*h*nc;

    float *gradientx = new float[(size_t)w*h*nc];
    float *gradienty = new float[(size_t)w*h*nc];
    float *divergence = new float[(size_t)w*h*nc];
    float *Laplacian = new float[(size_t)w*h*nc];


    float *d_imgIn = NULL;
    float *d_gradientx = NULL;
    float *d_gradienty = NULL;
    float *d_divergence = NULL;
    float *d_Laplacian = NULL;

    size_t nbytes = (size_t)(len)*sizeof(float);
    cudaMalloc(&d_imgIn, nbytes); CUDA_CHECK;
    cudaMalloc(&d_gradientx, nbytes); CUDA_CHECK;
    cudaMalloc(&d_gradienty, nbytes); CUDA_CHECK;
    cudaMalloc(&d_divergence, nbytes); CUDA_CHECK;
    cudaMalloc(&d_Laplacian, nbytes); CUDA_CHECK;


    cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice);

    dim3 block = dim3(32,16,nc);
    dim3 grid = dim3(len/block.x+1, 1, 1);

    getGradient<<<grid, block>>>(d_gradientx, d_gradienty, d_imgIn, w, h, nc);
    getDivergence<<<grid, block>>>(d_divergence, d_imgIn, w, h, nc);
    getLaplacian<<<grid, block>>>(d_Laplacian, d_imgIn, w, h, nc);

    cudaMemcpy(gradientx, d_gradientx, nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(gradienty, d_gradienty, nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(divergence, d_divergence, nbytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(Laplacian, d_Laplacian, nbytes, cudaMemcpyDeviceToHost);

    cudaFree(d_imgIn);
    cudaFree(d_gradientx);
    cudaFree(d_gradienty);
    cudaFree(d_divergence);
    cudaFree(d_Laplacian);
    timer.end();
    t = timer.get();  // elapsed time in seconds
    cout << "time4: " << t*1000 << " ms" << endl;




    //#############################################
    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed


    convert_layered_to_mat(Igradientx, gradientx);
    convert_layered_to_mat(Igradienty, gradienty);
    //convert_layered_to_mat(divergence, divergence);
    convert_layered_to_mat(ILaplacian, Laplacian);

    showImage("gradientx", Igradientx, 100, 100);
    showImage("gradienty", Igradienty, 100, 100);
    showImage("Laplacian", ILaplacian, 100, 100);
    //showImage("gradientx", gradientx, 100, 100);


#ifdef CAMERA
    // end of camera loop
    }
#else
    // wait for key inputs
    cv::waitKey(0);
#endif




    // save input and result
    cv::imwrite("image_input.png",mIn*255.f);  // "imwrite" assumes channel range [0,255]
    cv::imwrite("image_result.png",mOut*255.f);

    // free allocated arrays
    delete[] imgIn;
    delete[] imgOut;

    delete[] gradientx;
    delete[] gradienty;
    delete[] divergence;
    delete[] Laplacian;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



