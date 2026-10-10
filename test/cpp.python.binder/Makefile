PYTHON_CONFIG ?= python3-config
CXX ?= g++
CXXFLAGS ?= -O2 -Wall -Wextra -std=c++17
PY_CFLAGS := $(shell $(PYTHON_CONFIG) --includes)
PY_LDFLAGS := $(shell $(PYTHON_CONFIG) --embed --ldflags 2>/dev/null || $(PYTHON_CONFIG) --ldflags)

TARGET := cpp_python_binder_demo
SRC := cpp.python.binder.cc

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $(PY_CFLAGS) $< -o $@ $(PY_LDFLAGS)

clean:
	rm -f $(TARGET)

.PHONY: all clean
