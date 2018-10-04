CXX = g++
SOURCE_FILES = filterbank.cpp device.cpp 

CFLAGS = -Wall -std=c++11 
LIBS = -lrt -lOpenCL 
INCLUDES = -Iinclude -Iinclude/catch2

SRC_DIR := src
BUILD_DIR := build
TEST_DIR := test
OBJS := $(SOURCE_FILES:.cpp=.o)

SRCS := $(addprefix $(SRC_DIR)/, $(SOURCE_FILES))
OBJS := $(addprefix $(BUILD_DIR)/, $(OBJS))

DAS := prun -np 1 -native '-C TitanX --gres=gpu:1' 
#DAS = 

make: $(OBJS) $(SRCS)
	$(CXX) $(CFLAGS) $(LIBS) $(INCLUDES) $(OBJS) $(SRC_DIR)/main.cpp -o fflagger
	$(CXX) $(CFLAGS) $(LIBS) $(INCLUDES) $(OBJS) $(SRC_DIR)/sim_main.cpp -o sim_fflagger
	$(DAS) ./sim_fflagger -i ~/real.fil -o ~/del.fil

test: $(OBJS) $(SRCS)
	$(CXX) $(CFLAGS) $(LIBS) $(INCLUDES) $(OBJS) $(TEST_DIR)/tests.cpp -o $(BUILD_DIR)/tests
	$(DAS) $(BUILD_DIR)/tests [transpose]

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CFLAGS) $(LIBS) $(INCLUDES) -c $<  -o $@


clean:
	rm build/* fflagger

	


