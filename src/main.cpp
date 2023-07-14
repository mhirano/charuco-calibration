#include <iostream>
#include "CharucoCalibrator.h"

int main(int argc, char **argv) {

    auto calibrator = new CharucoCalibrator();

    /*
     * 単眼キャリブレーション
     */

//    calibrator->calibrateMono("../data/avsleftcam", "../data/avsleftcam-result");
    calibrator->estimateBoardPose("../data/avsleftcam", "../data/avsleftcam-result");

    /*
     * ステレオキャリブレーション
     */
//    calibrator->calibrateStereo("../data/calibImgs/stereoImgs", "../data/calibImgs/resultStereo");

    return 0;
}