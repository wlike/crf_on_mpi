//
//  CRF++ -- Yet Another CRF toolkit
//
//  $Id: common.h 1588 2007-02-12 09:03:39Z taku $;
//
//  Copyright(C) 2005-2007 Taku Kudo <taku@chasen.org>
//
#ifndef CRFPP_COMMON_H_
#define CRFPP_COMMON_H_

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <cmath>

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#define NOMINMAX
#include <windows.h>
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
# define OUTPUT_MODE std::ios::binary|std::ios::out
#else
# define OUTPUT_MODE std::ios::out
#endif

#define COPYRIGHT  "CRF++: Yet Another CRF Tool Kit\nCopyright (C) "    \
  "2005-2013 Taku Kudo, All rights reserved.\n"
#define MODEL_VERSION 100
#define BUF_SIZE 8192
#define EPS (1E-6)

namespace CRFPP {
// helper functions defined in the paper
inline double sgn(double x) {
  if (x > 0) return 1.0;
  else if (x < 0) return -1.0;
  return 0.0;
}

inline bool zero(double x) {
  return (fabs(x) <= EPS);
}

inline size_t getCpuCount() {
  size_t result = 1;
#if defined(_WIN32) && !defined(__CYGWIN__)
  SYSTEM_INFO si;
  ::GetSystemInfo(&si);
  result = si.dwNumberOfProcessors;
#else
#ifdef HAVE_SYS_CONF_SC_NPROCESSORS_CONF
  const long n = sysconf(_SC_NPROCESSORS_CONF);
  if (n == -1) {
    return 1;
  }
  result = static_cast<size_t>(n);
#endif
#endif
  return result;
}

inline unsigned short getThreadSize(unsigned short size) {
  if (size == 0) {
    return static_cast<unsigned short>(getCpuCount());
  }
  return size;
}

inline bool toLower(std::string *s) {
  for (size_t i = 0; i < s->size(); ++i) {
    char c = (*s)[i];
    if ((c >= 'A') && (c <= 'Z')) {
      c += 'a' - 'A';
      (*s)[i] = c;
    }
  }
  return true;
}

template <class Iterator>
inline size_t tokenizeCSV(char *str,
                          Iterator out, size_t max) {
  char *eos = str + std::strlen(str);
  char *start = 0;
  char *end = 0;
  size_t n = 0;

  for (; str < eos; ++str) {
    while (*str == ' ' || *str == '\t') ++str;  // skip white spaces
    bool inquote = false;
    if (*str == '"') {
      start = ++str;
      end = start;
      for (; str < eos; ++str) {
        if (*str == '"') {
          str++;
          if (*str != '"')
            break;
        }
        *end++ = *str;
      }
      inquote = true;
      str = std::find(str, eos, ',');
    } else {
      start = str;
      str = std::find(str, eos, ',');
      end = str;
    }
    if (max-- > 1) *end = '\0';
    *out++ = start;
    ++n;
    if (max == 0) break;
  }

  return n;
}

template <class Iterator>
inline size_t tokenize(char *str, const char *del,
                       Iterator out, size_t max) {
  char *stre = str + std::strlen(str);
  const char *dele = del + std::strlen(del);
  size_t size = 0;

  while (size < max) {
    char *n = std::find_first_of(str, stre, del, dele);
    *n = '\0';
    *out++ = str;
    ++size;
    if (n == stre) break;
    str = n + 1;
  }

  return size;
}

// continus run of space is regarded as one space
template <class Iterator>
inline size_t tokenize2(char *str, const char *del,
                        Iterator out, size_t max) {
  char *stre = str + std::strlen(str);
  const char *dele = del + std::strlen(del);
  size_t size = 0;

  while (size < max) {
    char *n = std::find_first_of(str, stre, del, dele);
    *n = '\0';
    if (*str != '\0') {
      *out++ = str;
      ++size;
    }
    if (n == stre) break;
    str = n + 1;
  }

  return size;
}

inline void dtoa(double val, char *s) {
  std::sprintf(s, "%-16f", val);
  char *p = s;
  for (; *p != ' '; ++p) {}
  *p = '\0';
  return;
}

template <class T> inline void itoa(T val, char *s) {
  char *t;
  T mod;

  if (val < 0) {
    *s++ = '-';
    val = -val;
  }
  t = s;

  while (val) {
    mod = val % 10;
    *t++ = static_cast<char>(mod)+ '0';
    val /= 10;
  }

  if (s == t) *t++ = '0';
  *t = '\0';
  std::reverse(s, t);

  return;
}

template <class T>
inline void uitoa(T val, char *s) {
  char *t;
  T mod;
  t = s;

  while (val) {
    mod = val % 10;
    *t++ = static_cast<char>(mod) + '0';
    val /= 10;
  }

  if (s == t) *t++ = '0';
  *t = '\0';
  std::reverse(s, t);

  return;
}

#if defined(_WIN32) && !defined(__CYGWIN__)
std::wstring Utf8ToWide(const std::string &input);
std::string WideToUtf8(const std::wstring &input);
#endif

#if defined(_WIN32) && !defined(__CYGWIN__)
#define WPATH(path) (CRFPP::Utf8ToWide(path).c_str())
#else
#define WPATH(path) (path)
#endif

#define _ITOA(_n) do {                          \
    char buf[64];                               \
    itoa(_n, buf);                              \
    append(buf);                                \
    return *this; } while (0)

#define _UITOA(_n) do {                         \
    char buf[64];                               \
    uitoa(_n, buf);                             \
    append(buf);                                \
    return *this; } while (0)

#define _DTOA(_n) do {                          \
    char buf[64];                               \
    dtoa(_n, buf);                              \
    append(buf);                                \
    return *this; } while (0)

class string_buffer: public std::string {
 public:
  string_buffer& operator<<(double _n)             { _DTOA(_n); }
  string_buffer& operator<<(short int _n)          { _ITOA(_n); }
  string_buffer& operator<<(int _n)                { _ITOA(_n); }
  string_buffer& operator<<(long int _n)           { _ITOA(_n); }
  string_buffer& operator<<(unsigned short int _n) { _UITOA(_n); }
  string_buffer& operator<<(unsigned int _n)       { _UITOA(_n); }
  string_buffer& operator<<(unsigned long int _n)  { _UITOA(_n); }
  string_buffer& operator<<(char _n) {
    push_back(_n);
    return *this;
  }
  string_buffer& operator<<(const char* _n) {
    append(_n);
    return *this;
  }
  string_buffer& operator<<(const std::string& _n) {
    append(_n);
    return *this;
  }
};

class die {
 public:
  die() {}
  ~die() {
    std::cerr << std::endl;
    exit(-1);
  }
  int operator&(std::ostream&) { return 0; }
};

struct whatlog {
  std::ostringstream stream_;
  std::string str_;
  const char *str() {
    str_ = stream_.str();
    return str_.c_str();
  }
};

class wlog {
 public:
  wlog(whatlog *what) : what_(what) {
    what_->stream_.clear();
  }
  bool operator&(std::ostream &) {
    return false;
  }
 private:
  whatlog *what_;
};
}  // namespace CRFPP

#define WHAT what_.stream_

#define CHECK_FALSE(condition) \
 if (condition) {} else return \
   wlog(&what_) & what_.stream_ <<              \
      __FILE__ << "(" << __LINE__ << ") [" << #condition << "] "

#define CHECK_DIE(condition) \
(condition) ? 0 : die() & std::cerr << __FILE__ << \
"(" << __LINE__ << ") [" << #condition << "] "

#ifdef USE_MPI
#include <stdint.h>
#include <map>
struct WorkerInfo {
    uint8_t data_part_id;
    uint32_t feature_function_num;
    // local_id => <global_id, function_num>
    std::map<uint32_t, std::pair<uint32_t, uint8_t> > ids_map;
};
#endif  // USE_MPI

#endif
