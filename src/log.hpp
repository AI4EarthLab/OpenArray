#ifndef __LOG_HPP__
#define __LOG_HPP__

#include <iostream>
#include <boost/log/common.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/attributes/timer.hpp>
#include <boost/log/attributes/named_scope.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/support/date_time.hpp>


//namespace logging = boost::log;
//namespace sinks = boost::log::sinks;
//namespace attrs = boost::log::attributes;
//namespace src = boost::log::sources;
//namespace expr = boost::log::expressions;
//namespace keywords = boost::log::keywords;


using boost::shared_ptr;

namespace oa {
  namespace logging{


enum severity_level
{
    normal,
    notification,
    warning,
    error,
    critical
};

template< typename CharT, typename TraitsT >
inline std::basic_ostream< CharT, TraitsT >& operator<< (
    std::basic_ostream< CharT, TraitsT >& strm, severity_level lvl){
         static const char* const str[] =
         {
            "normal",
            "notification",
            "warning",
            "error",
            "critical"
        };
        if (static_cast< std::size_t >(lvl) < (sizeof(str) / sizeof(*str)))
            strm << str[lvl];
        else
            strm << static_cast< int >(lvl);
        return strm;
    }

extern void logging_start(int);

extern void write_log(int);

extern void write_log_error(int, std::string);
  }//namespace
}//namespcae

#endif
