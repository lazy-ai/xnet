/* Created on 2017-12-03
 * Author: Binbin Zhang
 */

#include <fstream>

#include "xnet.h"

void XNet::Read(std::string model_file) {
    NetProto net_proto;
    std::fstream in(model_file, std::ios::in | std::ios::binary);
    if (!in) {
        ERROR("file %s does not exist", model_file.c_str());
    } else {
        net_proto.ParseFromIstream(&in);
    }
    
    for (int i = 0; i < net_proto.nodes_size(); i++) {
        const NodeProto &node = net_proto.nodes(i);
        CHECK(node.has_name());
        std::cout << node.name() << " " << node.node_type() << std::endl;
        switch (node.node_type()) {
            case NodeProto::FULLY_CONNECT:
                break;
            case NodeProto::QUANTIZE_FULLY_CONNECT:
                break;
            case NodeProto::RELU:
                break;
            case NodeProto::SIGMOID:
                break;
            case NodeProto::TANH:
                break;
            case NodeProto::SOFTMAX:
                break;
            default:
                ERROR("unknown node type %d", node.node_type());
        }
    }
}

