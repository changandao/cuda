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
using namespace std;

// uncomment to use the camera
//#define CAMERA


//Exercise 7
float* getKernel(float sigma, int radius, const char *kname)
{
    int len = (size_t)radius*2+1;
    float *k = new float[len*len];
    if(kname == "Gaus"){
        float sum = 0;
        float tmp = 0;
        for(int j = 0; j < len; j++)
        {
            for(int i = 0; i < len; i++)
            {
                tmp = expf(-(powf((i-radius),2.f) + powf(j-radius, 2.f))/(2.f*powf(sigma,2.f)))/(2.f*3.1416f*powf(sigma,2.f));
                sum = sum + tmp;
                k[i + (size_t)j*len] = tmp;
            }
        }
        float upscale = 1.f/sum;
        for(int ind = 0;ind < len*len; ind++) k[ind] = k[ind] * upscale;
    }else if(kname == "Gradx"){
        k[0] = -3.f/32.f;
        k[1] = 0.f;
        k[2] = 3.f/32.f;
        k[3] = -10.f/32.f;
        k[4] = 0.f;
        k[5] = 10.f/32.f;
        k[6] = -3.f/32.f;
        k[7] = 0.f;
        k[8] = 3.f/32.f;
    }
    else if(kname == "Grady"){
        k[0] = -3.f/32.f;
        k[3] = 0.f;
        k[6] = 3.f/32.f;
        k[1] = -10.f/32.f;
        k[4] = 0.f;
        k[7] = 10.f/32.f;
        k[2] = -3.f/32.f;
        k[5] = 0.f;
        k[8] = 3.f/32.f;
    }
    return k;
}



__global__ void Gconv2(float *imgO, const float *imgI, const float *kernel, int w, int h, int nc, int r)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;

    int kw = (size_t)2*r + 1;
    float value = 0.f;
    if(x<w && y<h) {
        for (int c = 0; c < nc; c++) {
            float sum = 0.f;
            size_t ind = x + (size_t) y * w + (size_t) w * h * c;
            for (int kj = -r; kj < r; kj++) {
                for (int ki = -r; ki < r; ki++) {
                    int kii = min(max(0, x+ki),w-1);
                    int kjj = min(max(0, y+kj),h-1);
                    value = imgI[kii + (size_t)(kjj*w) + (size_t)w*h*c];
                    sum += value * kernel[(r - ki) + (r - kj) * kw];
                }
            }
            imgO[ind] = sum;
        }
    }
}


__global__ void robGrad(float *gradientx, float *gradienty, const float *imgI, const float *kernelx, const float *kernely, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    int r = 1;
    int kw = (size_t)2*r + 1;
    float value = 0.f;
    if(x<w && y<h) {
        for (int c = 0; c < nc; c++) {
            float sumx = 0.f;
            float sumy = 0.f;
            size_t ind = x + (size_t) y * w + (size_t) w * h * c;
            for (int kj = -r; kj < r; kj++) {
                for (int ki = -r; ki < r; ki++) {
                    int kii = min(max(0, x+ki), w-1);
                    int kjj = min(max(0, y+kj), h-1);
                    value = imgI[kii + (size_t)(kjj*w) + (size_t)w*h*c];
                    sumx += value * kernelx[(r + ki) + (r + kj) * kw];
                    sumy += value * kernely[(r + ki) + (r + kj) * kw];
                }
            }
            gradientx[ind] = sumx;
            gradienty[ind] = sumy;
        }
    }
}


__global__ void getM(float *m11, float *m12, float *m22, float *gradientx, float *gradienty, int w, int h, int nc)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    float sum11 = 0.f;
    float sum12 = 0.f;
    float sum22 = 0.f;
    if(x<w && y<h) {
        size_t ind = x + (size_t) y * w;
        for (int c = 0; c < nc; c++) {
            size_t indc = x + (size_t) y * w + (size_t)w*h*c;
            sum11 += gradientx[indc] * gradientx[indc];
            sum12 += gradientx[indc] * gradienty[indc];
            sum22 += gradienty[indc] * gradienty[indc];
        }
        m11[ind] = sum11;
        m12[ind] = sum12;
        m22[ind] = sum22;
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


    Timer timer; timer.start();
    // ###
    // ###
    // ### TODO: Main computation
    // ###
    // ###

    float sigma = 2.1;
    int r = ceil(3.f * sigma);
    int len = w*h*nc;

    //##

    float *k = getKernel(sigma, r, "Gaus");
    float *dkx = getKernel(sigma, r, "Gradx");
    float *dky = getKernel(sigma, r, "Grady");

    float *convimO = new float[(size_t)w*h*nc];
    float *m11 = new float[(size_t)w*h*nc];
    float *m12 = new float[(size_t)w*h*nc];
    float *m22 = new float[(size_t)w*h*nc];

    cv::Mat convImg(h,w,mIn.type());
    cv::Mat Im11(h,w,mIn.type());
    cv::Mat Im12(h,w,mIn.type());
    cv::Mat Im22(h,w,mIn.type());



    cout<< "GPU is running!!!!";
    float *d_k = NULL;
    float *d_imgIn = NULL;
    float *d_kx = NULL;
    float *d_ky = NULL;

    float *d_m11 = NULL;
    float *d_m12 = NULL;
    float *d_m22 = NULL;
    float *d_T11 = NULL;
    float *d_T12 = NULL;
    float *d_T22 = NULL;

    float *d_convimO = NULL;
    float *d_gradientx = NULL;
    float *d_gradienty = NULL;

    size_t nbytes = (size_t)(len)*sizeof(float);
    cudaMalloc(&d_k, nbytes); CUDA_CHECK;
    cudaMalloc(&d_kx, nbytes); CUDA_CHECK;
    cudaMalloc(&d_ky, nbytes); CUDA_CHECK;
    cudaMalloc(&d_imgIn, nbytes); CUDA_CHECK;

    cudaMalloc(&d_convimO, nbytes); CUDA_CHECK;
    cudaMalloc(&d_gradientx, nbytes); CUDA_CHECK;
    cudaMalloc(&d_gradienty, nbytes); CUDA_CHECK;
    cudaMalloc(&d_m11, nbytes); CUDA_CHECK;
    cudaMalloc(&d_m12, nbytes); CUDA_CHECK;
    cudaMalloc(&d_m22, nbytes); CUDA_CHECK;
    cudaMalloc(&d_T11, nbytes); CUDA_CHECK;
    cudaMalloc(&d_T12, nbytes); CUDA_CHECK;
    cudaMalloc(&d_T22, nbytes); CUDA_CHECK;


    cudaMemcpy(d_k, k, nbytes, cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMemcpy(d_kx, dkx, nbytes, cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMemcpy(d_ky, dky, nbytes, cudaMemcpyHostToDevice);CUDA_CHECK;
    cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice);CUDA_CHECK;

    dim3 block = dim3(32,32,1);
    dim3 grid = dim3((block.x + w-1)/block.x, (block.y + h-1)/block.y, 1);

    Gconv2<<<grid, block>>>(d_convimO, d_imgIn, d_k, w, h, nc, r);CUDA_CHECK;

    robGrad<<<grid, block>>>(d_gradientx, d_gradienty, d_convimO, d_kx, d_ky, w, h, nc);CUDA_CHECK;

    getM<<<grid, block>>>(d_m11, d_m12, d_m12, d_gradientx, d_gradienty, w,h,nc);CUDA_CHECK;

    Gconv2<<<grid, block>>>(d_T11, d_m11, d_k, w, h, nc, r);CUDA_CHECK;
    Gconv2<<<grid, block>>>(d_T12, d_m12, d_k, w, h, nc, r);CUDA_CHECK;
    Gconv2<<<grid, block>>>(d_T22, d_m22, d_k, w, h, nc, r);CUDA_CHECK;


    cudaMemcpy(convimO, d_convimO, nbytes, cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(convimO, d_gradientx, nbytes, cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(convimO, d_gradienty, nbytes, cudaMemcpyDeviceToHost);CUDA_CHECK;

    cudaMemcpy(m11, d_m11, nbytes, cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(m22, d_m11, nbytes, cudaMemcpyDeviceToHost);CUDA_CHECK;
    cudaMemcpy(m22, d_m12, nbytes, cudaMemcpyDeviceToHost);CUDA_CHECK;


    cudaFree(d_k);CUDA_CHECK;
    cudaFree(d_kx);CUDA_CHECK;
    cudaFree(d_ky);CUDA_CHECK;

    cudaFree(d_convimO);CUDA_CHECK;
    cudaFree(d_imgIn);CUDA_CHECK;
    cudaFree(d_m11);CUDA_CHECK;
    cudaFree(d_m12);CUDA_CHECK;
    cudaFree(d_m22);CUDA_CHECK;
    cudaFree(d_T11);CUDA_CHECK;
    cudaFree(d_T12);CUDA_CHECK;
    cudaFree(d_T22);CUDA_CHECK;

    cudaFree(d_gradientx);CUDA_CHECK;
    cudaFree(d_gradienty);CUDA_CHECK;



    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

    convert_layered_to_mat(convImg, convimO);
    showImage("convolution CPU", convImg, 100, 100);

    convert_layered_to_mat(Im11, m11);
    convert_layered_to_mat(Im12, m12);
    convert_layered_to_mat(Im22, m22);

    int scaleup = 1.f;

    Im11 *= scaleup;
    Im12 *= scaleup;
    Im22 *= scaleup;
    showImage("m11", Im11, 100, 100);
    showImage("m12", Im12, 100, 100);
    showImage("m22", Im22, 100, 100);



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
    delete[] k;
    delete[] dkx;
    delete[] dky;
    delete[] convimO;

    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



