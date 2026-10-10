#include <Python.h>

#include <iostream>
#include <filesystem>
#include <string>
#include <vector>

namespace {

bool check_and_print_py_error(const char* context) {
  if (!PyErr_Occurred()) {
    return false;
  }
  std::cerr << "[python error] " << context << std::endl;
  PyErr_Print();
  return true;
}

PyObject* import_module(const char* module_name, const char* module_dir) {
  PyObject* sys_path = PySys_GetObject("path");
  if (!sys_path) {
    std::cerr << "Failed to get sys.path" << std::endl;
    return nullptr;
  }

  PyObject* path = PyUnicode_FromString(module_dir);
  if (!path) {
    check_and_print_py_error("PyUnicode_FromString(module_dir)");
    return nullptr;
  }

  if (PyList_Append(sys_path, path) != 0) {
    Py_DECREF(path);
    check_and_print_py_error("PyList_Append(sys.path)");
    return nullptr;
  }
  Py_DECREF(path);

  PyObject* name = PyUnicode_FromString(module_name);
  if (!name) {
    check_and_print_py_error("PyUnicode_FromString(module_name)");
    return nullptr;
  }

  PyObject* module = PyImport_Import(name);
  Py_DECREF(name);
  if (!module) {
    check_and_print_py_error("PyImport_Import");
    return nullptr;
  }
  return module;
}

long call_add_ints(PyObject* module, long a, long b) {
  PyObject* func = PyObject_GetAttrString(module, "add_ints");
  if (!func || !PyCallable_Check(func)) {
    Py_XDECREF(func);
    check_and_print_py_error("add_ints lookup");
    return 0;
  }

  PyObject* args = PyTuple_Pack(2, PyLong_FromLong(a), PyLong_FromLong(b));
  PyObject* result = PyObject_CallObject(func, args);
  Py_DECREF(args);
  Py_DECREF(func);

  if (!result) {
    check_and_print_py_error("add_ints call");
    return 0;
  }

  long value = PyLong_AsLong(result);
  Py_DECREF(result);
  if (check_and_print_py_error("PyLong_AsLong")) {
    return 0;
  }
  return value;
}

std::string call_format_person(PyObject* module,
                               const std::string& name,
                               int age,
                               double height) {
  PyObject* func = PyObject_GetAttrString(module, "format_person");
  if (!func || !PyCallable_Check(func)) {
    Py_XDECREF(func);
    check_and_print_py_error("format_person lookup");
    return {};
  }

  PyObject* args = PyTuple_Pack(3,
                                PyUnicode_FromString(name.c_str()),
                                PyLong_FromLong(age),
                                PyFloat_FromDouble(height));
  PyObject* result = PyObject_CallObject(func, args);
  Py_DECREF(args);
  Py_DECREF(func);

  if (!result) {
    check_and_print_py_error("format_person call");
    return {};
  }

  PyObject* utf8 = PyUnicode_AsUTF8String(result);
  Py_DECREF(result);
  if (!utf8) {
    check_and_print_py_error("PyUnicode_AsUTF8String");
    return {};
  }

  std::string value = PyBytes_AsString(utf8);
  Py_DECREF(utf8);
  return value;
}

double call_weighted_sum(PyObject* module,
                          const std::vector<double>& values,
                          const std::vector<double>& weights) {
  PyObject* func = PyObject_GetAttrString(module, "weighted_sum");
  if (!func || !PyCallable_Check(func)) {
    Py_XDECREF(func);
    check_and_print_py_error("weighted_sum lookup");
    return 0.0;
  }

  PyObject* list_values = PyList_New(static_cast<Py_ssize_t>(values.size()));
  PyObject* list_weights = PyList_New(static_cast<Py_ssize_t>(weights.size()));
  if (!list_values || !list_weights) {
    Py_XDECREF(list_values);
    Py_XDECREF(list_weights);
    Py_DECREF(func);
    check_and_print_py_error("PyList_New");
    return 0.0;
  }

  for (size_t i = 0; i < values.size(); ++i) {
    PyList_SetItem(list_values,
                   static_cast<Py_ssize_t>(i),
                   PyFloat_FromDouble(values[i]));
  }
  for (size_t i = 0; i < weights.size(); ++i) {
    PyList_SetItem(list_weights,
                   static_cast<Py_ssize_t>(i),
                   PyFloat_FromDouble(weights[i]));
  }

  PyObject* args = PyTuple_Pack(2, list_values, list_weights);
  Py_DECREF(list_values);
  Py_DECREF(list_weights);

  PyObject* result = PyObject_CallObject(func, args);
  Py_DECREF(args);
  Py_DECREF(func);

  if (!result) {
    check_and_print_py_error("weighted_sum call");
    return 0.0;
  }

  double value = PyFloat_AsDouble(result);
  Py_DECREF(result);
  if (check_and_print_py_error("PyFloat_AsDouble")) {
    return 0.0;
  }
  return value;
}

}  // namespace

int main() {
  Py_Initialize();

  std::filesystem::path module_dir = std::filesystem::current_path();
  if (!std::filesystem::exists(module_dir / "py_funcs.py")) {
    module_dir /= "test/cpp.python.binder";
  }
  if (!std::filesystem::exists(module_dir / "py_funcs.py")) {
    std::cerr << "py_funcs.py not found in current directory or test/cpp.python.binder"
              << std::endl;
    Py_Finalize();
    return 1;
  }

  PyObject* module = import_module("py_funcs", module_dir.string().c_str());
  if (!module) {
    Py_Finalize();
    return 1;
  }

  long sum = call_add_ints(module, 7, 35);
  std::string profile = call_format_person(module, "Dora", 28, 1.72);
  double weighted = call_weighted_sum(module, {1.0, 2.0, 3.0}, {0.2, 0.3, 0.5});

  std::cout << "add_ints result: " << sum << std::endl;
  std::cout << "format_person result: " << profile << std::endl;
  std::cout << "weighted_sum result: " << weighted << std::endl;

  Py_DECREF(module);
  Py_Finalize();
  return 0;
}
