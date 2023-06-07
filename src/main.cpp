#include <iostream>
#include <thread>
#include <opencv2/opencv.hpp>

#include "InterThreadMessenger.hpp"

struct DispMsg : public MsgData {
    cv::Mat image;
};

void workerThread(InterThreadMessenger<DispMsg>& messenger)
{
    cv::Mat lena = cv::imread("../data/lena.png", cv::IMREAD_GRAYSCALE);
    cv::Mat temp;

    while (1) {
        /**
         * messengerへの送信用メッセージへのポインタを取得
         */
        DispMsg *md = messenger.prepareMsg();

        /**
         * メイン処理開始
         * 送信用メッセージの中のデータ(md->image)に対して書き込みを行う．
         */
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        int k = ceil(rand()%5)*4+1;
        cv::GaussianBlur( lena, md->image, cv::Size(k,k), 0 );
        std::cout << "Image processing for a single frame done ..." << std::endl;
        /* * * * * * * * * * * * メイン処理ここまで * * * * * * * * * * * */

        /**
         * messenger に対して送信を実行
         */
        messenger.send();

        /**
         * messenger が終了している場合はworkerも終了
         */
        if (messenger.isClosed()) {
            break;
        }
    }
}


int main(int argc, char **argv) {
    /**
     * 処理スレッドと描画スレッド間でデータを受け渡すクラスを用意
     * DispMsg は描画する画像を保持する構造体
     */
    InterThreadMessenger<DispMsg> messenger;

    /**
     * 処理スレッド (workerThread) に messenger を渡して起動
     * messengerを渡すことにより，描画
     */
    std::thread thr_work(workerThread, std::ref(messenger));

    cv::namedWindow("display", cv::WINDOW_AUTOSIZE);

    /**
     * messengerが保持しているDispMsgから画像を読み込んで表示する．
     * 描画用データに更新がなければ何もしない
     */
    while (1) {
        DispMsg *md = messenger.receive();
        if (md != nullptr) {
            cv::imshow("disp", md->image);
            std::cout << "Display result image ..." << std::endl;
        }
        if (cv::waitKey(30) > 0) {
            break;
        }
    }

    /**
     * messengerをcloseすることで，workerThreadの処理も終了させる．
     */
    messenger.close();

    /**
     * workerThreadが終了可能となり次第，終了する．
     */
    if (thr_work.joinable()) thr_work.join();

    return 0;
}
