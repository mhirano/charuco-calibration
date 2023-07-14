//
// Created by Masahiro Hirano <masahiro.dll@gmail.com>
//

#include <filesystem>

struct CharucoCalibrator{

    /**
     * ChArUcoボードを使った単眼カメラのキャリブレーション．
     * @return
     */
	bool calibrateMono(std::string calibDataPathStr, std::string calibResultPathStr);

    /**
     * ChArUcoボードを使ったステレオカメラのキャリブレーション．
     * 内部的に，各カメラの内部パラメータのキャリブレーションも行う．
     * @return
     */
    bool calibrateStereo(std::string calibDataPathStr, std::string calibResultPathStr);

    bool estimateBoardPose(std::string calibDataPathStr, std::string calibResultPathStr);

};