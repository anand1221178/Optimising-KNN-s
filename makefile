# Compiler
CXX = g++

# Compiler flags
# -std=c++11: Use C++11 standard
# -Wall -Wextra: Enable most warnings
# -O2: Optimization level 2 
# -fopenmp: Enable OpenMP support 
CXXFLAGS = -std=c++11 -Wall -Wextra -fopenmp


# Target executable name
TARGET = knn

# Source files
SRCS = knn.cpp

# Default target (what runs when you just type "make")
all: $(TARGET)

# Rule to link the executable
$(TARGET): $(SRCS)
	$(CXX) $(CXXFLAGS) $(SRCS) -o $(TARGET) # $(LDFLAGS) # Add LDFLAGS if needed

# Rule to clean up generated files
clean:
	rm -f $(TARGET)

# Phony targets (targets that don't represent actual files)
.PHONY: all clean
