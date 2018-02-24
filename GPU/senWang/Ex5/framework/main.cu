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
#include <stdio.h>
using namespace std;

// uncomment to use the camera
//#define CAMERA

//Exercise 5
float* gausKernel(float sigma, int radius)
{
    int len = (size_t)radius*2+1;
    float *k = new float[len*len];
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
    return k;
}


void scale(float *output, float *input,int len)
{
    float maximum = 0.f;
    for(int i = 0;i<len;i++)
    {
        if (maximum < input[i]) maximum = input[i];

    }
    float upscale = 1.f/maximum;
    for(int j = 0;j < len; j++)
    {
        output[j] = output[j] * upscale;
    }
}



// CPU-version convolution
void convolution(float *imgO, const float *imgI, const float *kernel, int w, int h, int nc, int kw)
{
    size_t ind = 0;
    int r = floor(kw/2);
    float value = 0.f;
    for(int c = 0; c < nc; c++)
    {
        for(int j = 0;j < h; j++)
        {
            for(int i = 0; i < w; i++)
            {
                float sum = 0.f;
                ind = i + (size_t)j*w + (size_t)w*h*c;
                for(int kj = -r; kj < r; kj++)
                {
                    for(int ki = -r; ki < r; ki++)
                    {
                        int kii = min(max(0, i+ki),w-1);
                        int kjj = min(max(0, j+kj),h-1);
//                        if(i+ki<0)
//                        {
//                            value = imgI[(size_t)j*w + (size_t)w*h*c];
//                        }
//                        else if(i+ki > w)
//                        {
//                            value = imgI[w-1+(size_t)j*w + (size_t)w*h*c];
//                        }
//                        else
//                        {
//                            if(j+kj < 0) value = imgI[i + (size_t)w*h*c];
//                            else if((j+kj > h)) value = imgI[i + (size_t)(h-1)*w + (size_t)w*h*c];
//                            else value = imgI[i+ki+(size_t)(j+kj)*w + (size_t)w*h*c];
//                        }
                        value = imgI[kii + (size_t)(kjj*w) + (size_t)w*h*c];
                        sum += value * kernel[(r-ki)+(r-kj)*kw];
                    }
                }
                imgO[ind] = sum;
            }
        }
    }
}

//GPU_version convolution

__global__ void Gconvolution(float *imgO, const float *imgI, const float *kernel, int w, int h, int nc, int kw)
{
    int x = threadIdx.x + blockDim.x * blockIdx.x;
    int y = threadIdx.y + blockDim.y * blockIdx.y;
    if(x==4 && y==5)printf("%s \n", "A string");
    int r = kw/2;
    float value = 0.f;
    if(x<w && y<h) {
        for (int c = 0; c < nc; c++) {
            float sum = 0.f;
            size_t ind = x + (size_t) y * w + (size_t) w * h * c;
            for (int kj = -r; kj < r; kj++) {
                for (int ki = -r; ki < r; ki++) {
//                    if (x + ki < 0) {
//                        value = imgI[(size_t) y * w + (size_t) w * h * c];
//                    } else if (x + ki > w) {
//                        value = imgI[w - 1 + (size_t) y * w + (size_t) w * h * c];
//                    } else {
//                        if (y + kj < 0) value = imgI[x + (size_t) w * h * c];
//                        else if ((y + kj > h)) value = imgI[x + (size_t)(h - 1) * w + (size_t) w * h * c];
//                        else value = imgI[x + ki + (size_t)(y + kj) * w + (size_t) w * h * c];
//                    }
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
    int GPU = 0;
    getParam("GPU", GPU, argc, argv);
    cout << "GPU: " << GPU << endl;


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

    int len = w*h*nc;
    int version = GPU;// GPU =1 is on, otherwise it's off
    Timer timer; timer.start();

    //Exercise 5

    float sigma = 2.1;
    int r = ceil(3.f * sigma);
    int sizeofk = 2 * r + 1;
    int lenofA = sizeofk * sizeofk;

    //##
    float *k = gausKernel(sigma, r);
//    float *k_dach = k;
//    scale(k_dach, k_dach, lenofA);

    //##
    float *convimO = new float[(size_t)w*h*nc];

    if(version == 0)convolution(convimO,imgIn,k,w,h,nc,sizeofk);//CPU version
    else{
        //GPU version
        cout<< "GPU is running!!!!";
        float *d_k = NULL;
        float *d_convimO = NULL;
        float *d_imgIn = NULL;

        size_t nbytes = (size_t)(len)*sizeof(float);
        cudaMalloc(&d_k, nbytes); CUDA_CHECK;
        cudaMalloc(&d_convimO, nbytes); CUDA_CHECK;
        cudaMalloc(&d_imgIn, nbytes); CUDA_CHECK;

        cudaMemcpy(d_k, k, nbytes, cudaMemcpyHostToDevice);CUDA_CHECK;
        cudaMemcpy(d_imgIn, imgIn, nbytes, cudaMemcpyHostToDevice);CUDA_CHECK;

        dim3 block = dim3(32,32,1);
        dim3 grid = dim3((block.x + w-1)/block.x, (block.y + h-1)/block.y, 1);
        Gconvolution<<<grid, block>>>(d_convimO, d_imgIn, d_k, w, h, nc, sizeofk);CUDA_CHECK;
        cudaMemcpy(convimO, d_convimO, nbytes, cudaMemcpyDeviceToHost);CUDA_CHECK;

        cudaFree(d_k);CUDA_CHECK;
        cudaFree(d_convimO);CUDA_CHECK;
        cudaFree(d_imgIn);CUDA_CHECK;
    }

    cout << endl;
    timer.end();
    int t = timer.get();  // elapsed time in seconds

    cout << "time5: " << t*1000 << " ms" << endl;

//    cv::Mat kern(sizeofk, sizeofk, mIn.type());
//    convert_layered_to_mat(kern, k_dach);


    //#############################################
    // show input image
    showImage("Input", mIn, 100, 100);  // show at position (x_from_left=100,y_from_above=100)

    // show output image: first convert to interleaved opencv format from the layered raw array
    convert_layered_to_mat(mOut, imgOut);
    showImage("Output", mOut, 100+w+40, 100);

    // ### Display your own output images here as needed

    //showImage("Gauskernel", kernel, 100, 100);

    cv::Mat convImg(h,w,mIn.type());

    convert_layered_to_mat(convImg, convimO);
    showImage("convolution CPU", convImg, 100, 100);

    cv::Mat convImg(h,w,mIn.type());

    convert_layered_to_mat(convImg, convimO);
    showImage("convolution CPU", convImg, 100, 100);

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

    delete[] imgIn;
    delete[] imgOut;

    delete[] k;
    //delete[] k_dach;
    delete[] convimO;


    // close all opencv windows
    cvDestroyAllWindows();
    return 0;
}



