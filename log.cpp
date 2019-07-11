#ifndef __LOG_CPP__
#define __LOG_CPP__
#ifndef SUNWAY
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
#include "log.hpp"

//namespace logging = boost::log;
//namespace sinks = boost::log::sinks;
//namespace attrs = boost::log::attributes;
//namespace src = boost::log::sources;
//namespace expr = boost::log::expressions;
//namespace keywords = boost::log::keywords;


using boost::shared_ptr;

namespace oa {
  namespace logging{

extern void logging_start(int world_rank){
     boost::log::add_console_log(std::clog, boost::log::keywords::format = "%TimeStamp%: %Message%");
     boost::log::add_file_log
      (
        "monitor_in_thread_"+std::to_string(world_rank)+".log",
        boost::log::keywords::filter = boost::log::expressions::attr< severity_level >("Severity") >= warning,
        boost::log::keywords::format = boost::log::expressions::stream
            << boost::log::expressions::format_date_time< boost::posix_time::ptime >("TimeStamp", "%Y-%m-%d, %H:%M:%S.%f")
            << " [" << boost::log::expressions::format_date_time< boost::log::attributes::timer::value_type >("Uptime", "%O:%M:%S")
            << "] [" << boost::log::expressions::format_named_scope("Scope", boost::log::keywords::format = "%n (%f:%l)")
            << "] <" << boost::log::expressions::attr< severity_level >("Severity")
            << "> " << boost::log::expressions::message

     );

    boost::log::add_common_attributes();
    boost::log::core::get()->add_thread_attribute("Scope", boost::log::attributes::named_scope());

    BOOST_LOG_FUNCTION();
}

extern void write_log(int world_rank){
    boost::log::sources::logger lg;
    BOOST_LOG(lg) << "Hello, World!";
    boost::log::sources::severity_logger< severity_level > slg;
    slg.add_attribute("Uptime", boost::log::attributes::timer());

    BOOST_LOG_SEV(slg, normal) <<"Thread_"+std::to_string(world_rank)+": A normal severity message, will not pass to the file";
    BOOST_LOG_SEV(slg, warning) <<"Thread_"+std::to_string(world_rank)+": A warning severity message, will pass to the file";
    BOOST_LOG_SEV(slg, error) << "Thread_"+std::to_string(world_rank)+": An error severity message, will pass to the file";
}

extern void write_log_error(int world_rank, std::string function_name){
    boost::log::sources::severity_logger< severity_level > slg;
    slg.add_attribute("Uptime", boost::log::attributes::timer());
std::cout<<function_name<<std::endl;
    BOOST_LOG_SEV(slg, error) << "Thread_"+std::to_string(world_rank)+": An error severity message,in the funcion of "+function_name;
}


}
}
#endif
#endif
