/* Created on 2017-12-03
 * Author: Binbin Zhang
 */

#include "xnet.h"

int main(int argc, char *argv[]) {
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    XNet xnet;
    xnet.Read(argv[1]);
    google::protobuf::ShutdownProtobufLibrary();

    xnet.Info();
    return 0;
}


