#include <opencv2/opencv.hpp>
using namespace cv;

int main() {
    Mat ImgL = imread("sample/01L.png", IMREAD_GRAYSCALE);
    Mat ImgR = imread("sample/01R.png", IMREAD_GRAYSCALE);

    Ptr<StereoBM> stereo = StereoBM::create(16*4, 21);

    Mat diff;
    stereo->compute(ImgL, ImgR, diff);

    normalize(diff, diff, 0, 255, NORM_MINMAX, CV_8U);

    imshow("Depth Map", diff);
    waitKey(0);

    return 0;
}
