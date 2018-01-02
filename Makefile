CXX = g++

CXXFLAGS = -g -std=c++11 -I . -lprotobuf -lopenblas -lpthread -msse4.1 -D USE_BLAS # -D QUANTIZE_BIAS

OBJ = xnet.o tensor.o net.pb.o

TEST = test/mnist-test

BIN = tools/xnet-read tools/xnet-quantization

all: $(TEST) $(BIN) $(OBJ)

test/%: test/%.cc $(OBJ)
	$(CXX) $< $(OBJ) $(CXXFLAGS) -o $@

tools/%: tools/%.cc $(OBJ)
	$(CXX) $< $(OBJ) $(CXXFLAGS) -o $@

proto: 
	protoc -I=. --cpp_out=. net.proto
	protoc -I=. --python_out=./tools net.proto

xnet.o: xnet.h net.pb.h utils.h 
tensor.o: tensor.h

.PHONY: clean

clean:
	rm -rf $(OBJ); rm -rf $(TEST); rm -rf $(BIN)

