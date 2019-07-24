/*
 * Operator.cpp
 * evaluate the expression graph
 *
=======================================================*/

#include "Operator.hpp"
#include "utils/utils.hpp"
#include "utils/calcTime.hpp"
#include "Kernel.hpp"
#ifdef _OSX_
#include "Jit_Driver.osx.hpp"
#else
#include "Jit_Driver.hpp"
#endif
#include "Grid.hpp"
#include <fstream>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>
#include <dlfcn.h>
#include "Diagnosis.hpp"
#ifndef SUNWAY
#include <boost/format.hpp>
#endif

using namespace oa::kernel;

namespace oa {
  namespace ops{

    // needs to set all attributes to the new node
    NodePtr new_node(const ArrayPtr &ap) {
      NodePtr np = NodePool::global()->get();
      np->set_type(TYPE_DATA);
      np->set_data(ap);
      np->set_data_type(ap->get_data_type());
      np->set_shape(ap->shape());
      np->set_scalar(ap->is_scalar());
      np->set_seqs(ap->is_seqs());
      np->set_pos(ap->get_pos());
      np->set_bitset(ap->get_bitset());
      np->set_pseudo(ap->is_pseudo());
      //np->set_data_list_size(1);

      return np;
    }

    // only operator min_max & rep will call this function
    // other binary operator will call new_node_type in modules
    NodePtr new_node(NodeType type, NodePtr u, NodePtr v){
      NodePtr np = NodePool::global()->get();
      np->set_type(type);
      np->add_input(0, u);
      np->add_input(1, v);
      np->set_lbound({{0, 0, 0}});
      np->set_rbound({{0, 0, 0}});
      np->set_update();
      np->set_data_type(u->get_data_type());
      
      if(u->get_pos() != -1)
        np->set_pos(u->get_pos());
      else if(v->get_pos() != -1)
        np->set_pos(v->get_pos());
     
      return np;
    }

    const NodeDesc& get_node_desc(NodeType type){

      static bool has_init = false;                                            
      static OpDescList s;
      
      // use NodeType.fypp to initialize the NodeDesc
      if (!has_init) {
        s.resize(NUM_NODE_TYPES);
        //intialize node descriptions.
        

        s[TYPE_UNKNOWN].type = TYPE_UNKNOWN;
        s[TYPE_UNKNOWN].name = "unknown";
        s[TYPE_UNKNOWN].sy = "slice";
        s[TYPE_UNKNOWN].ew = true;
        s[TYPE_UNKNOWN].cl = false;
        s[TYPE_UNKNOWN].expr = "";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_UNKNOWN].func = NULL;
        
        ///!:else
        ///s[TYPE_UNKNOWN].func = NULL;
        ///!:endif
        
        s[TYPE_UNKNOWN].rt = 0;


        

        s[TYPE_DATA].type = TYPE_DATA;
        s[TYPE_DATA].name = "data";
        s[TYPE_DATA].sy = "";
        s[TYPE_DATA].ew = true;
        s[TYPE_DATA].cl = false;
        s[TYPE_DATA].expr = "";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DATA].func = NULL;
        
        ///!:else
        ///s[TYPE_DATA].func = NULL;
        ///!:endif
        
        s[TYPE_DATA].rt = 0;


        

        s[TYPE_REF].type = TYPE_REF;
        s[TYPE_REF].name = "ref";
        s[TYPE_REF].sy = "";
        s[TYPE_REF].ew = false;
        s[TYPE_REF].cl = false;
        s[TYPE_REF].expr = "";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_REF].func = NULL;
        
        ///!:else
        ///s[TYPE_REF].func = NULL;
        ///!:endif
        
        s[TYPE_REF].rt = 0;


        

        s[TYPE_PLUS].type = TYPE_PLUS;
        s[TYPE_PLUS].name = "plus";
        s[TYPE_PLUS].sy = "+";
        s[TYPE_PLUS].ew = true;
        s[TYPE_PLUS].cl = false;
        s[TYPE_PLUS].expr = "A+B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_PLUS].func = kernel_plus;
        
        ///!:else
        ///s[TYPE_PLUS].func = NULL;
        ///!:endif
        
        s[TYPE_PLUS].rt = 0;


        

        s[TYPE_MINUS].type = TYPE_MINUS;
        s[TYPE_MINUS].name = "minus";
        s[TYPE_MINUS].sy = "-";
        s[TYPE_MINUS].ew = true;
        s[TYPE_MINUS].cl = false;
        s[TYPE_MINUS].expr = "A-B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_MINUS].func = kernel_minus;
        
        ///!:else
        ///s[TYPE_MINUS].func = NULL;
        ///!:endif
        
        s[TYPE_MINUS].rt = 0;


        

        s[TYPE_MULT].type = TYPE_MULT;
        s[TYPE_MULT].name = "mult";
        s[TYPE_MULT].sy = "*";
        s[TYPE_MULT].ew = true;
        s[TYPE_MULT].cl = false;
        s[TYPE_MULT].expr = "A*B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_MULT].func = kernel_mult;
        
        ///!:else
        ///s[TYPE_MULT].func = NULL;
        ///!:endif
        
        s[TYPE_MULT].rt = 0;


        

        s[TYPE_DIVD].type = TYPE_DIVD;
        s[TYPE_DIVD].name = "divd";
        s[TYPE_DIVD].sy = "/";
        s[TYPE_DIVD].ew = true;
        s[TYPE_DIVD].cl = false;
        s[TYPE_DIVD].expr = "A/B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DIVD].func = kernel_divd;
        
        ///!:else
        ///s[TYPE_DIVD].func = NULL;
        ///!:endif
        
        s[TYPE_DIVD].rt = 0;


        

        s[TYPE_GT].type = TYPE_GT;
        s[TYPE_GT].name = "gt";
        s[TYPE_GT].sy = ">";
        s[TYPE_GT].ew = false;
        s[TYPE_GT].cl = false;
        s[TYPE_GT].expr = "A>B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_GT].func = kernel_gt;
        
        ///!:else
        ///s[TYPE_GT].func = NULL;
        ///!:endif
        
        s[TYPE_GT].rt = 0;


        

        s[TYPE_GE].type = TYPE_GE;
        s[TYPE_GE].name = "ge";
        s[TYPE_GE].sy = ">=";
        s[TYPE_GE].ew = false;
        s[TYPE_GE].cl = false;
        s[TYPE_GE].expr = "A>=B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_GE].func = kernel_ge;
        
        ///!:else
        ///s[TYPE_GE].func = NULL;
        ///!:endif
        
        s[TYPE_GE].rt = 0;


        

        s[TYPE_LT].type = TYPE_LT;
        s[TYPE_LT].name = "lt";
        s[TYPE_LT].sy = "<";
        s[TYPE_LT].ew = false;
        s[TYPE_LT].cl = false;
        s[TYPE_LT].expr = "A<B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_LT].func = kernel_lt;
        
        ///!:else
        ///s[TYPE_LT].func = NULL;
        ///!:endif
        
        s[TYPE_LT].rt = 0;


        

        s[TYPE_LE].type = TYPE_LE;
        s[TYPE_LE].name = "le";
        s[TYPE_LE].sy = "<=";
        s[TYPE_LE].ew = false;
        s[TYPE_LE].cl = false;
        s[TYPE_LE].expr = "A<=B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_LE].func = kernel_le;
        
        ///!:else
        ///s[TYPE_LE].func = NULL;
        ///!:endif
        
        s[TYPE_LE].rt = 0;


        

        s[TYPE_EQ].type = TYPE_EQ;
        s[TYPE_EQ].name = "eq";
        s[TYPE_EQ].sy = "==";
        s[TYPE_EQ].ew = false;
        s[TYPE_EQ].cl = false;
        s[TYPE_EQ].expr = "A==B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_EQ].func = kernel_eq;
        
        ///!:else
        ///s[TYPE_EQ].func = NULL;
        ///!:endif
        
        s[TYPE_EQ].rt = 0;


        

        s[TYPE_NE].type = TYPE_NE;
        s[TYPE_NE].name = "ne";
        s[TYPE_NE].sy = "!=";
        s[TYPE_NE].ew = false;
        s[TYPE_NE].cl = false;
        s[TYPE_NE].expr = "A/=B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_NE].func = kernel_ne;
        
        ///!:else
        ///s[TYPE_NE].func = NULL;
        ///!:endif
        
        s[TYPE_NE].rt = 0;


        

        s[TYPE_MIN].type = TYPE_MIN;
        s[TYPE_MIN].name = "min";
        s[TYPE_MIN].sy = "min";
        s[TYPE_MIN].ew = false;
        s[TYPE_MIN].cl = false;
        s[TYPE_MIN].expr = "min";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_MIN].func = kernel_min;
        
        ///!:else
        ///s[TYPE_MIN].func = NULL;
        ///!:endif
        
        s[TYPE_MIN].rt = 0;


        

        s[TYPE_MAX].type = TYPE_MAX;
        s[TYPE_MAX].name = "max";
        s[TYPE_MAX].sy = "max";
        s[TYPE_MAX].ew = false;
        s[TYPE_MAX].cl = false;
        s[TYPE_MAX].expr = "max";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_MAX].func = kernel_max;
        
        ///!:else
        ///s[TYPE_MAX].func = NULL;
        ///!:endif
        
        s[TYPE_MAX].rt = 0;


        

        s[TYPE_MIN_AT].type = TYPE_MIN_AT;
        s[TYPE_MIN_AT].name = "min_at";
        s[TYPE_MIN_AT].sy = "min";
        s[TYPE_MIN_AT].ew = false;
        s[TYPE_MIN_AT].cl = false;
        s[TYPE_MIN_AT].expr = "min_at";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_MIN_AT].func = kernel_min_at;
        
        ///!:else
        ///s[TYPE_MIN_AT].func = NULL;
        ///!:endif
        
        s[TYPE_MIN_AT].rt = 0;


        

        s[TYPE_MAX_AT].type = TYPE_MAX_AT;
        s[TYPE_MAX_AT].name = "max_at";
        s[TYPE_MAX_AT].sy = "max";
        s[TYPE_MAX_AT].ew = false;
        s[TYPE_MAX_AT].cl = false;
        s[TYPE_MAX_AT].expr = "max_at";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_MAX_AT].func = kernel_max_at;
        
        ///!:else
        ///s[TYPE_MAX_AT].func = NULL;
        ///!:endif
        
        s[TYPE_MAX_AT].rt = 0;


        

        s[TYPE_ABS_MAX].type = TYPE_ABS_MAX;
        s[TYPE_ABS_MAX].name = "abs_max";
        s[TYPE_ABS_MAX].sy = "abs_max";
        s[TYPE_ABS_MAX].ew = false;
        s[TYPE_ABS_MAX].cl = false;
        s[TYPE_ABS_MAX].expr = "";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_ABS_MAX].func = kernel_abs_max;
        
        ///!:else
        ///s[TYPE_ABS_MAX].func = NULL;
        ///!:endif
        
        s[TYPE_ABS_MAX].rt = 0;


        

        s[TYPE_ABS_MIN].type = TYPE_ABS_MIN;
        s[TYPE_ABS_MIN].name = "abs_min";
        s[TYPE_ABS_MIN].sy = "abs_min";
        s[TYPE_ABS_MIN].ew = false;
        s[TYPE_ABS_MIN].cl = false;
        s[TYPE_ABS_MIN].expr = "";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_ABS_MIN].func = kernel_abs_min;
        
        ///!:else
        ///s[TYPE_ABS_MIN].func = NULL;
        ///!:endif
        
        s[TYPE_ABS_MIN].rt = 0;


        

        s[TYPE_ABS_MAX_AT].type = TYPE_ABS_MAX_AT;
        s[TYPE_ABS_MAX_AT].name = "abs_max_at";
        s[TYPE_ABS_MAX_AT].sy = "abs_max";
        s[TYPE_ABS_MAX_AT].ew = false;
        s[TYPE_ABS_MAX_AT].cl = false;
        s[TYPE_ABS_MAX_AT].expr = "";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_ABS_MAX_AT].func = kernel_abs_max_at;
        
        ///!:else
        ///s[TYPE_ABS_MAX_AT].func = NULL;
        ///!:endif
        
        s[TYPE_ABS_MAX_AT].rt = 0;


        

        s[TYPE_ABS_MIN_AT].type = TYPE_ABS_MIN_AT;
        s[TYPE_ABS_MIN_AT].name = "abs_min_at";
        s[TYPE_ABS_MIN_AT].sy = "abs_min";
        s[TYPE_ABS_MIN_AT].ew = false;
        s[TYPE_ABS_MIN_AT].cl = false;
        s[TYPE_ABS_MIN_AT].expr = "";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_ABS_MIN_AT].func = kernel_abs_min_at;
        
        ///!:else
        ///s[TYPE_ABS_MIN_AT].func = NULL;
        ///!:endif
        
        s[TYPE_ABS_MIN_AT].rt = 0;


        

        s[TYPE_MIN2].type = TYPE_MIN2;
        s[TYPE_MIN2].name = "min2";
        s[TYPE_MIN2].sy = "min2";
        s[TYPE_MIN2].ew = false;
        s[TYPE_MIN2].cl = false;
        s[TYPE_MIN2].expr = "min";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_MIN2].func = kernel_min2;
        
        ///!:else
        ///s[TYPE_MIN2].func = NULL;
        ///!:endif
        
        s[TYPE_MIN2].rt = 0;


        

        s[TYPE_MAX2].type = TYPE_MAX2;
        s[TYPE_MAX2].name = "max2";
        s[TYPE_MAX2].sy = "max2";
        s[TYPE_MAX2].ew = false;
        s[TYPE_MAX2].cl = false;
        s[TYPE_MAX2].expr = "max";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_MAX2].func = kernel_max2;
        
        ///!:else
        ///s[TYPE_MAX2].func = NULL;
        ///!:endif
        
        s[TYPE_MAX2].rt = 0;


        

        s[TYPE_POW].type = TYPE_POW;
        s[TYPE_POW].name = "pow";
        s[TYPE_POW].sy = "pow";
        s[TYPE_POW].ew = true;
        s[TYPE_POW].cl = false;
        s[TYPE_POW].expr = "(A)**M";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_POW].func = kernel_pow;
        
        ///!:else
        ///s[TYPE_POW].func = NULL;
        ///!:endif
        
        s[TYPE_POW].rt = 2;


        

        s[TYPE_EXP].type = TYPE_EXP;
        s[TYPE_EXP].name = "exp";
        s[TYPE_EXP].sy = "exp";
        s[TYPE_EXP].ew = true;
        s[TYPE_EXP].cl = false;
        s[TYPE_EXP].expr = "exp(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_EXP].func = kernel_exp;
        
        ///!:else
        ///s[TYPE_EXP].func = NULL;
        ///!:endif
        
        s[TYPE_EXP].rt = 2;


        

        s[TYPE_SIN].type = TYPE_SIN;
        s[TYPE_SIN].name = "sin";
        s[TYPE_SIN].sy = "sin";
        s[TYPE_SIN].ew = true;
        s[TYPE_SIN].cl = false;
        s[TYPE_SIN].expr = "sin(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_SIN].func = kernel_sin;
        
        ///!:else
        ///s[TYPE_SIN].func = NULL;
        ///!:endif
        
        s[TYPE_SIN].rt = 2;


        

        s[TYPE_TAN].type = TYPE_TAN;
        s[TYPE_TAN].name = "tan";
        s[TYPE_TAN].sy = "tan";
        s[TYPE_TAN].ew = true;
        s[TYPE_TAN].cl = false;
        s[TYPE_TAN].expr = "tan(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_TAN].func = kernel_tan;
        
        ///!:else
        ///s[TYPE_TAN].func = NULL;
        ///!:endif
        
        s[TYPE_TAN].rt = 2;


        

        s[TYPE_COS].type = TYPE_COS;
        s[TYPE_COS].name = "cos";
        s[TYPE_COS].sy = "cos";
        s[TYPE_COS].ew = true;
        s[TYPE_COS].cl = false;
        s[TYPE_COS].expr = "cos(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_COS].func = kernel_cos;
        
        ///!:else
        ///s[TYPE_COS].func = NULL;
        ///!:endif
        
        s[TYPE_COS].rt = 2;


        

        s[TYPE_RCP].type = TYPE_RCP;
        s[TYPE_RCP].name = "rcp";
        s[TYPE_RCP].sy = "rcp";
        s[TYPE_RCP].ew = true;
        s[TYPE_RCP].cl = false;
        s[TYPE_RCP].expr = "1.0/A";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_RCP].func = kernel_rcp;
        
        ///!:else
        ///s[TYPE_RCP].func = NULL;
        ///!:endif
        
        s[TYPE_RCP].rt = 2;


        

        s[TYPE_SQRT].type = TYPE_SQRT;
        s[TYPE_SQRT].name = "sqrt";
        s[TYPE_SQRT].sy = "sqrt";
        s[TYPE_SQRT].ew = true;
        s[TYPE_SQRT].cl = false;
        s[TYPE_SQRT].expr = "sqrt(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_SQRT].func = kernel_sqrt;
        
        ///!:else
        ///s[TYPE_SQRT].func = NULL;
        ///!:endif
        
        s[TYPE_SQRT].rt = 2;


        

        s[TYPE_ASIN].type = TYPE_ASIN;
        s[TYPE_ASIN].name = "asin";
        s[TYPE_ASIN].sy = "asin";
        s[TYPE_ASIN].ew = true;
        s[TYPE_ASIN].cl = false;
        s[TYPE_ASIN].expr = "asin(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_ASIN].func = kernel_asin;
        
        ///!:else
        ///s[TYPE_ASIN].func = NULL;
        ///!:endif
        
        s[TYPE_ASIN].rt = 2;


        

        s[TYPE_ACOS].type = TYPE_ACOS;
        s[TYPE_ACOS].name = "acos";
        s[TYPE_ACOS].sy = "acos";
        s[TYPE_ACOS].ew = true;
        s[TYPE_ACOS].cl = false;
        s[TYPE_ACOS].expr = "acos(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_ACOS].func = kernel_acos;
        
        ///!:else
        ///s[TYPE_ACOS].func = NULL;
        ///!:endif
        
        s[TYPE_ACOS].rt = 2;


        

        s[TYPE_ATAN].type = TYPE_ATAN;
        s[TYPE_ATAN].name = "atan";
        s[TYPE_ATAN].sy = "atan";
        s[TYPE_ATAN].ew = true;
        s[TYPE_ATAN].cl = false;
        s[TYPE_ATAN].expr = "atan(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_ATAN].func = kernel_atan;
        
        ///!:else
        ///s[TYPE_ATAN].func = NULL;
        ///!:endif
        
        s[TYPE_ATAN].rt = 2;


        

        s[TYPE_ABS].type = TYPE_ABS;
        s[TYPE_ABS].name = "abs";
        s[TYPE_ABS].sy = "abs";
        s[TYPE_ABS].ew = true;
        s[TYPE_ABS].cl = false;
        s[TYPE_ABS].expr = "abs(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_ABS].func = kernel_abs;
        
        ///!:else
        ///s[TYPE_ABS].func = NULL;
        ///!:endif
        
        s[TYPE_ABS].rt = 2;


        

        s[TYPE_LOG].type = TYPE_LOG;
        s[TYPE_LOG].name = "log";
        s[TYPE_LOG].sy = "log";
        s[TYPE_LOG].ew = true;
        s[TYPE_LOG].cl = false;
        s[TYPE_LOG].expr = "log(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_LOG].func = kernel_log;
        
        ///!:else
        ///s[TYPE_LOG].func = NULL;
        ///!:endif
        
        s[TYPE_LOG].rt = 2;


        

        s[TYPE_UPLUS].type = TYPE_UPLUS;
        s[TYPE_UPLUS].name = "uplus";
        s[TYPE_UPLUS].sy = "+";
        s[TYPE_UPLUS].ew = true;
        s[TYPE_UPLUS].cl = false;
        s[TYPE_UPLUS].expr = "+(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_UPLUS].func = kernel_uplus;
        
        ///!:else
        ///s[TYPE_UPLUS].func = NULL;
        ///!:endif
        
        s[TYPE_UPLUS].rt = 2;


        

        s[TYPE_UMINUS].type = TYPE_UMINUS;
        s[TYPE_UMINUS].name = "uminus";
        s[TYPE_UMINUS].sy = "-";
        s[TYPE_UMINUS].ew = true;
        s[TYPE_UMINUS].cl = false;
        s[TYPE_UMINUS].expr = "-(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_UMINUS].func = kernel_uminus;
        
        ///!:else
        ///s[TYPE_UMINUS].func = NULL;
        ///!:endif
        
        s[TYPE_UMINUS].rt = 2;


        

        s[TYPE_LOG10].type = TYPE_LOG10;
        s[TYPE_LOG10].name = "log10";
        s[TYPE_LOG10].sy = "log10";
        s[TYPE_LOG10].ew = true;
        s[TYPE_LOG10].cl = false;
        s[TYPE_LOG10].expr = "log10(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_LOG10].func = kernel_log10;
        
        ///!:else
        ///s[TYPE_LOG10].func = NULL;
        ///!:endif
        
        s[TYPE_LOG10].rt = 2;


        

        s[TYPE_TANH].type = TYPE_TANH;
        s[TYPE_TANH].name = "tanh";
        s[TYPE_TANH].sy = "tanh";
        s[TYPE_TANH].ew = true;
        s[TYPE_TANH].cl = false;
        s[TYPE_TANH].expr = "tanh(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_TANH].func = kernel_tanh;
        
        ///!:else
        ///s[TYPE_TANH].func = NULL;
        ///!:endif
        
        s[TYPE_TANH].rt = 2;


        

        s[TYPE_SINH].type = TYPE_SINH;
        s[TYPE_SINH].name = "sinh";
        s[TYPE_SINH].sy = "sinh";
        s[TYPE_SINH].ew = true;
        s[TYPE_SINH].cl = false;
        s[TYPE_SINH].expr = "sinh(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_SINH].func = kernel_sinh;
        
        ///!:else
        ///s[TYPE_SINH].func = NULL;
        ///!:endif
        
        s[TYPE_SINH].rt = 2;


        

        s[TYPE_COSH].type = TYPE_COSH;
        s[TYPE_COSH].name = "cosh";
        s[TYPE_COSH].sy = "cosh";
        s[TYPE_COSH].ew = true;
        s[TYPE_COSH].cl = false;
        s[TYPE_COSH].expr = "cosh(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_COSH].func = kernel_cosh;
        
        ///!:else
        ///s[TYPE_COSH].func = NULL;
        ///!:endif
        
        s[TYPE_COSH].rt = 2;


        

        s[TYPE_DXC].type = TYPE_DXC;
        s[TYPE_DXC].name = "dxc";
        s[TYPE_DXC].sy = "dxc";
        s[TYPE_DXC].ew = true;
        s[TYPE_DXC].cl = false;
        s[TYPE_DXC].expr = "DXC(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DXC].func = kernel_dxc;
        
        ///!:else
        ///s[TYPE_DXC].func = NULL;
        ///!:endif
        
        s[TYPE_DXC].rt = 2;


        

        s[TYPE_DYC].type = TYPE_DYC;
        s[TYPE_DYC].name = "dyc";
        s[TYPE_DYC].sy = "dyc";
        s[TYPE_DYC].ew = true;
        s[TYPE_DYC].cl = false;
        s[TYPE_DYC].expr = "DYC(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DYC].func = kernel_dyc;
        
        ///!:else
        ///s[TYPE_DYC].func = NULL;
        ///!:endif
        
        s[TYPE_DYC].rt = 2;


        

        s[TYPE_DZC].type = TYPE_DZC;
        s[TYPE_DZC].name = "dzc";
        s[TYPE_DZC].sy = "dzc";
        s[TYPE_DZC].ew = true;
        s[TYPE_DZC].cl = false;
        s[TYPE_DZC].expr = "DZC(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DZC].func = kernel_dzc;
        
        ///!:else
        ///s[TYPE_DZC].func = NULL;
        ///!:endif
        
        s[TYPE_DZC].rt = 2;


        

        s[TYPE_AXB].type = TYPE_AXB;
        s[TYPE_AXB].name = "axb";
        s[TYPE_AXB].sy = "axb";
        s[TYPE_AXB].ew = true;
        s[TYPE_AXB].cl = false;
        s[TYPE_AXB].expr = "AXB(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_AXB].func = kernel_axb;
        
        ///!:else
        ///s[TYPE_AXB].func = NULL;
        ///!:endif
        
        s[TYPE_AXB].rt = 2;


        

        s[TYPE_AXF].type = TYPE_AXF;
        s[TYPE_AXF].name = "axf";
        s[TYPE_AXF].sy = "axf";
        s[TYPE_AXF].ew = true;
        s[TYPE_AXF].cl = false;
        s[TYPE_AXF].expr = "AXF(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_AXF].func = kernel_axf;
        
        ///!:else
        ///s[TYPE_AXF].func = NULL;
        ///!:endif
        
        s[TYPE_AXF].rt = 2;


        

        s[TYPE_AYB].type = TYPE_AYB;
        s[TYPE_AYB].name = "ayb";
        s[TYPE_AYB].sy = "ayb";
        s[TYPE_AYB].ew = true;
        s[TYPE_AYB].cl = false;
        s[TYPE_AYB].expr = "AYB(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_AYB].func = kernel_ayb;
        
        ///!:else
        ///s[TYPE_AYB].func = NULL;
        ///!:endif
        
        s[TYPE_AYB].rt = 2;


        

        s[TYPE_AYF].type = TYPE_AYF;
        s[TYPE_AYF].name = "ayf";
        s[TYPE_AYF].sy = "ayf";
        s[TYPE_AYF].ew = true;
        s[TYPE_AYF].cl = false;
        s[TYPE_AYF].expr = "AYF(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_AYF].func = kernel_ayf;
        
        ///!:else
        ///s[TYPE_AYF].func = NULL;
        ///!:endif
        
        s[TYPE_AYF].rt = 2;


        

        s[TYPE_AZB].type = TYPE_AZB;
        s[TYPE_AZB].name = "azb";
        s[TYPE_AZB].sy = "azb";
        s[TYPE_AZB].ew = true;
        s[TYPE_AZB].cl = false;
        s[TYPE_AZB].expr = "AZB(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_AZB].func = kernel_azb;
        
        ///!:else
        ///s[TYPE_AZB].func = NULL;
        ///!:endif
        
        s[TYPE_AZB].rt = 2;


        

        s[TYPE_AZF].type = TYPE_AZF;
        s[TYPE_AZF].name = "azf";
        s[TYPE_AZF].sy = "azf";
        s[TYPE_AZF].ew = true;
        s[TYPE_AZF].cl = false;
        s[TYPE_AZF].expr = "AZF(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_AZF].func = kernel_azf;
        
        ///!:else
        ///s[TYPE_AZF].func = NULL;
        ///!:endif
        
        s[TYPE_AZF].rt = 2;


        

        s[TYPE_DXB].type = TYPE_DXB;
        s[TYPE_DXB].name = "dxb";
        s[TYPE_DXB].sy = "dxb";
        s[TYPE_DXB].ew = true;
        s[TYPE_DXB].cl = false;
        s[TYPE_DXB].expr = "DXB(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DXB].func = kernel_dxb;
        
        ///!:else
        ///s[TYPE_DXB].func = NULL;
        ///!:endif
        
        s[TYPE_DXB].rt = 2;


        

        s[TYPE_DXF].type = TYPE_DXF;
        s[TYPE_DXF].name = "dxf";
        s[TYPE_DXF].sy = "dxf";
        s[TYPE_DXF].ew = true;
        s[TYPE_DXF].cl = false;
        s[TYPE_DXF].expr = "DXF(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DXF].func = kernel_dxf;
        
        ///!:else
        ///s[TYPE_DXF].func = NULL;
        ///!:endif
        
        s[TYPE_DXF].rt = 2;


        

        s[TYPE_DYB].type = TYPE_DYB;
        s[TYPE_DYB].name = "dyb";
        s[TYPE_DYB].sy = "dyb";
        s[TYPE_DYB].ew = true;
        s[TYPE_DYB].cl = false;
        s[TYPE_DYB].expr = "DYB(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DYB].func = kernel_dyb;
        
        ///!:else
        ///s[TYPE_DYB].func = NULL;
        ///!:endif
        
        s[TYPE_DYB].rt = 2;


        

        s[TYPE_DYF].type = TYPE_DYF;
        s[TYPE_DYF].name = "dyf";
        s[TYPE_DYF].sy = "dyf";
        s[TYPE_DYF].ew = true;
        s[TYPE_DYF].cl = false;
        s[TYPE_DYF].expr = "DYF(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DYF].func = kernel_dyf;
        
        ///!:else
        ///s[TYPE_DYF].func = NULL;
        ///!:endif
        
        s[TYPE_DYF].rt = 2;


        

        s[TYPE_DZB].type = TYPE_DZB;
        s[TYPE_DZB].name = "dzb";
        s[TYPE_DZB].sy = "dzb";
        s[TYPE_DZB].ew = true;
        s[TYPE_DZB].cl = false;
        s[TYPE_DZB].expr = "DZB(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DZB].func = kernel_dzb;
        
        ///!:else
        ///s[TYPE_DZB].func = NULL;
        ///!:endif
        
        s[TYPE_DZB].rt = 2;


        

        s[TYPE_DZF].type = TYPE_DZF;
        s[TYPE_DZF].name = "dzf";
        s[TYPE_DZF].sy = "dzf";
        s[TYPE_DZF].ew = true;
        s[TYPE_DZF].cl = false;
        s[TYPE_DZF].expr = "DZF(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_DZF].func = kernel_dzf;
        
        ///!:else
        ///s[TYPE_DZF].func = NULL;
        ///!:endif
        
        s[TYPE_DZF].rt = 2;


        

        s[TYPE_SUM].type = TYPE_SUM;
        s[TYPE_SUM].name = "sum";
        s[TYPE_SUM].sy = "sum";
        s[TYPE_SUM].ew = false;
        s[TYPE_SUM].cl = false;
        s[TYPE_SUM].expr = "sum";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_SUM].func = kernel_sum;
        
        ///!:else
        ///s[TYPE_SUM].func = NULL;
        ///!:endif
        
        s[TYPE_SUM].rt = 0;


        

        s[TYPE_CSUM].type = TYPE_CSUM;
        s[TYPE_CSUM].name = "csum";
        s[TYPE_CSUM].sy = "csum";
        s[TYPE_CSUM].ew = false;
        s[TYPE_CSUM].cl = false;
        s[TYPE_CSUM].expr = "csum(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_CSUM].func = kernel_csum;
        
        ///!:else
        ///s[TYPE_CSUM].func = NULL;
        ///!:endif
        
        s[TYPE_CSUM].rt = 0;


        

        s[TYPE_OR].type = TYPE_OR;
        s[TYPE_OR].name = "or";
        s[TYPE_OR].sy = "||";
        s[TYPE_OR].ew = false;
        s[TYPE_OR].cl = false;
        s[TYPE_OR].expr = "A.or.B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_OR].func = kernel_or;
        
        ///!:else
        ///s[TYPE_OR].func = NULL;
        ///!:endif
        
        s[TYPE_OR].rt = 0;


        

        s[TYPE_AND].type = TYPE_AND;
        s[TYPE_AND].name = "and";
        s[TYPE_AND].sy = "&&";
        s[TYPE_AND].ew = false;
        s[TYPE_AND].cl = false;
        s[TYPE_AND].expr = "A.and.B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_AND].func = kernel_and;
        
        ///!:else
        ///s[TYPE_AND].func = NULL;
        ///!:endif
        
        s[TYPE_AND].rt = 0;


        

        s[TYPE_NOT].type = TYPE_NOT;
        s[TYPE_NOT].name = "not";
        s[TYPE_NOT].sy = "!";
        s[TYPE_NOT].ew = false;
        s[TYPE_NOT].cl = false;
        s[TYPE_NOT].expr = ".not.B";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_NOT].func = kernel_not;
        
        ///!:else
        ///s[TYPE_NOT].func = NULL;
        ///!:endif
        
        s[TYPE_NOT].rt = 0;


        

        s[TYPE_REP].type = TYPE_REP;
        s[TYPE_REP].name = "rep";
        s[TYPE_REP].sy = "rep";
        s[TYPE_REP].ew = false;
        s[TYPE_REP].cl = true;
        s[TYPE_REP].expr = "rep(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_REP].func = kernel_rep;
        
        ///!:else
        ///s[TYPE_REP].func = NULL;
        ///!:endif
        
        s[TYPE_REP].rt = 0;


        

        s[TYPE_SHIFT].type = TYPE_SHIFT;
        s[TYPE_SHIFT].name = "shift";
        s[TYPE_SHIFT].sy = "shift";
        s[TYPE_SHIFT].ew = false;
        s[TYPE_SHIFT].cl = false;
        s[TYPE_SHIFT].expr = "shift(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_SHIFT].func = kernel_shift;
        
        ///!:else
        ///s[TYPE_SHIFT].func = NULL;
        ///!:endif
        
        s[TYPE_SHIFT].rt = 0;


        

        s[TYPE_CIRCSHIFT].type = TYPE_CIRCSHIFT;
        s[TYPE_CIRCSHIFT].name = "circshift";
        s[TYPE_CIRCSHIFT].sy = "circshift";
        s[TYPE_CIRCSHIFT].ew = false;
        s[TYPE_CIRCSHIFT].cl = false;
        s[TYPE_CIRCSHIFT].expr = "circshift(A)";
        
        
        ///!:if (('A' <= i[3] and i[3] <= 'F') or name == 'pow' or name == 'not')
        s[TYPE_CIRCSHIFT].func = kernel_circshift;
        
        ///!:else
        ///s[TYPE_CIRCSHIFT].func = NULL;
        ///!:endif
        
        s[TYPE_CIRCSHIFT].rt = 0;


        has_init = true;
      }  
      return s.at(type);
    }

    // write the expression graph into filename.dot
    void write_graph(const NodePtr& root, bool is_root,
            const char *filename) {
      if (MPI_RANK > 0) return ;
      static std::ofstream ofs;
      if (is_root) {
        ofs.open(filename);
        ofs<<"digraph G {"<<endl;
      }
      int id = root->get_id();
      ofs<<id;

      const NodeDesc & nd = get_node_desc(root->type());

      //char buffer[500];

      //sprintf(buffer, "[label=\"[%s]\\n id=%d \n (ref:%d) "
			// "\n (lb:%d %d %d) \n (rb: %d %d %d) \n (pseudo: %d) \n (up: %d)\"];",  
	    //nd.name, id, root.use_count(),
	    //root->get_lbound()[0], root->get_lbound()[1], root->get_lbound()[2],
	    //root->get_rbound()[0], root->get_rbound()[1], root->get_rbound()[2],
	    //root->is_pseudo(), root->need_update());

      //ofs<<buffer<<endl;
#ifndef SUNWAY
      ofs<<boost::format("[label=\"[%s]\\n id=%d \n (ref:%d) "
                          "\n (lb:%d %d %d) \n (rb: %d %d %d) \n (up: %d)\"];") 
        % nd.name % id % root.use_count()
        % root->get_lbound()[0] % root->get_lbound()[1] % root->get_lbound()[2]
        % root->get_rbound()[0] % root->get_rbound()[1] % root->get_rbound()[2] 
        % root->need_update() <<endl;
#endif
      for (int i = 0; i < root->input_size(); i++) {
        write_graph(root->input(i), false, filename);
        ofs<<id<<"->"<<root->input(i)->get_id()<<";"<<endl;
      }

      if (is_root) {
        ofs<<"}"<<endl;
        ofs.close();
      }
    }

    // force eval the expression graph, only use basic kernels
    ArrayPtr force_eval(NodePtr A) {
      if (A->has_data()) return A->get_data();

      vector<ArrayPtr> ops_ap;
      for (int i = 0; i < A->input_size(); i++) {
        ops_ap.push_back(force_eval(A->input(i)));
      }

      const NodeDesc& nd = get_node_desc(A->type());
      KernelPtr kernel_addr = nd.func;
      ArrayPtr ap = kernel_addr(ops_ap);
      ap->set_pseudo(A->is_pseudo());
      ap->set_bitset(A->get_bitset());
      ap->set_pos(A->get_pos());

      return ap;
    }

    // based on specific NodeType to change the lbound
    oa_int3 change_lbound(NodeType type, oa_int3 lb) {
      switch (type) {
        case TYPE_AXB:
        case TYPE_DXB:
        case TYPE_DXC:
          lb[0] = 1;
          break;
        case TYPE_AYB:
        case TYPE_DYB:
        case TYPE_DYC:
          lb[1] = 1;
          break;
        case TYPE_AZB:
        case TYPE_DZB:
        case TYPE_DZC:
          lb[2] = 1;
          break;
        default:
          break;
      }
      return lb;
    }

    // based on specific NodeType to change the rbound
    oa_int3 change_rbound(NodeType type, oa_int3 rb) {
      switch (type) {
        case TYPE_AXF:
        case TYPE_DXF:
        case TYPE_DXC:
          rb[0] = 1;
          break;
        case TYPE_AYF:
        case TYPE_DYF:
        case TYPE_DYC:
          rb[1] = 1;
          break;
        case TYPE_AZF:
        case TYPE_DZF:
        case TYPE_DZC:
          rb[2] = 1;
          break;
        default:
          break;
      }
      return rb;
    }

    //
    // =======================================================
    // to evaluate expression like A = B + C + D
    // we need to pass parameters to the fusion kernel
    // like: the data pointer to ans A, parameters B, C & D
    //       the shape of ans A, parameters B, C & D etc.
    // =======================================================
    //
    // NodePtr A :  the root of (sub)expression graph
    // list:        the data list which used in fusion kernel
    // update_list: the array list which needs to update boundary 
    // S:           the shape of array in data list which used in fusion kernel
    // ptr:         the final Partition pointer of ans
    // bt:          the final bitset of ans
    // lb_list:     the lbound list of array in data list which used in fusion kernel
    // rb_list:     the rbound list of array in data list which used in fusion kernel
    // lb_now:      the lbound from the root to the current node
    // rb_now:      the rbound from the root to the current node
    // data_list:   the data list of different shape, to check whether data has to transfer or not
    //
    void get_kernel_parameter_with_op(NodePtr A, vector<void*> &list, 
      vector<ArrayPtr> &update_list, vector<oa_int3> &S, PartitionPtr &ptr, 
      bitset<3> &bt, vector<oa_int3> &lb_list, vector<oa_int3> &rb_list,
      oa_int3 lb_now, oa_int3 rb_now, vector<ArrayPtr> &data_list) {
      bool find_in_data_list;
      ArrayPtr ap;
      // 1. the Node is a data node, put data into list
      if (A->has_data()) {
        ap = A->get_data();

        // 1.1 to check whether the ap needs to transfer or not
        if(!ap->is_scalar())
        {
          find_in_data_list = false;
          for(int i = 0; i < data_list.size(); i++){
            if(ap->shape() == data_list[i]->shape()){
              PartitionPtr pp = ap->get_partition();
              if(!(pp->equal(data_list[i]->get_partition()))){
                // ap has the same shape with data_list[i], but the partition is not the same
                ap = oa::funcs::transfer(ap, data_list[i]->get_partition());
              }
              find_in_data_list  = true;
              break;
            }
          }
          // it's the first time the shape appears
          if(!find_in_data_list) data_list.push_back(ap);
        }

        // 1.2 ap is a pseudo 3d, need to make_pseudo_3d
        if (ap->get_bitset() != bt && !ap->is_seqs_scalar() && ap->is_pseudo()) {
          if (ap->has_pseudo_3d() == false) {
            ap->set_pseudo_3d(oa::funcs::make_psudo3d(ap));
          }
          ap = ap->get_pseudo_3d();
        }

        // 1.3 put the array's data into list
        list.push_back(ap->get_buffer());

        // 1.4 determine the answer's partition
        if (ptr == NULL && ap->get_bitset() == bt) {
          ptr = ap->get_partition();
        }

        // 1.5 put the buffer shape into S which needs in fusion kernel
        // put the lb_now & rb_now into lb_list & rb_list which needs in update boundary
        if (!A->is_seqs_scalar()) {
          S.push_back(ap->buffer_shape());
          bool inupl = false;
          for(int i=0;i<update_list.size();i++){
            if(ap->get_buffer() == update_list[i]->get_buffer()){
              if(lb_list[i]==lb_now && rb_list[i]==rb_now){
                inupl =true;
                break;
              }
            }
          }
          //cout<<inupl<<endl;
          if(!inupl){
            update_list.push_back(ap);
            lb_list.push_back(lb_now);
            rb_list.push_back(rb_now);
          }
        }
        return ;
      }

      // 2. Operator node is not element wise, or need update 
      const NodeDesc &nd = get_node_desc(A->type());
      if (!nd.ew || A->need_update()) {
        // need change need_update's state in order to evaluate recursively
        bool flag = A->need_update();
        A->set_update(false);
        ArrayPtr ap = eval(A);
        A->set_update(flag);

        // 2.1 to check whether the ap needs to transfer or not
        if(!ap->is_scalar())
        {
          find_in_data_list = false;
          for(int i = 0; i < data_list.size(); i++){
            if(ap->shape() == data_list[i]->shape()){
              PartitionPtr pp = ap->get_partition();
              if(!(pp->equal(data_list[i]->get_partition()))){
                ap = oa::funcs::transfer(ap, data_list[i]->get_partition());
              }
              find_in_data_list  = true;
              break;
            }
          }
          if(!find_in_data_list) data_list.push_back(ap);
        }

        // 2.2 ap is a pseudo 3d, need to make_pseudo_3d
        if (ap->get_bitset() != bt && !ap->is_seqs_scalar() && ap->is_pseudo()) {
          if (ap->has_pseudo_3d() == false) {
            ap->set_pseudo_3d(oa::funcs::make_psudo3d(ap));
          }
          ap = ap->get_pseudo_3d();
        }

        // 2.3 put the array's data into list
        list.push_back(ap->get_buffer());

        // 2.4 determine the answer's partition
        if (ptr == NULL && ap->get_bitset() == bt) {
          ptr = ap->get_partition();
        }

        // 2.5 put the buffer shape into S which needs in fusion kernel
        // put the lb_now & rb_now into lb_list & rb_list which needs in update boundary
        if (!A->is_seqs_scalar()) {
          S.push_back(ap->buffer_shape());
          bool inupl = false;
          for(int i=0;i<update_list.size();i++){
            if(ap->get_buffer() == update_list[i]->get_buffer()){
              if(lb_list[i]==lb_now && rb_list[i]==rb_now){
                inupl =true;
                break;
              }
            }
          }
          //cout<<inupl<<endl;
          if(!inupl){
            update_list.push_back(ap);
            lb_list.push_back(lb_now);
            rb_list.push_back(rb_now);
          }

        }
        return ;
      }

      // 3. it's an operator node, get kernel parameters from it's child node 
      for (int i = 0; i < A->input_size(); i++) {
        get_kernel_parameter_with_op(A->input(i), list, update_list, S, ptr, bt,
            lb_list, rb_list, change_lbound(nd.type, lb_now), change_rbound(nd.type, rb_now), data_list);
      }

      // 4. if A is OPERATOR, need to bind grid if A.pos != -1
      if (A->input_size() == 1 && A->get_pos() != -1) {
        if (nd.type == TYPE_DXC ||
            nd.type == TYPE_DYC ||
            nd.type == TYPE_DZC ||
            nd.type == TYPE_DXB ||
            nd.type == TYPE_DXF ||
            nd.type == TYPE_DYB ||
            nd.type == TYPE_DYF ||
            nd.type == TYPE_DZB ||
            nd.type == TYPE_DZF) {

          // 4.1 get grid ptr
          ArrayPtr grid_ptr = Grid::global()->get_grid(A->get_pos(), nd.type);          
          // 4.2 get the grid's data into list
          list.push_back(grid_ptr->get_buffer());
          // 4.3 put the buffer shape into S
          S.push_back(grid_ptr->buffer_shape());
          //if (g_debug) grid_ptr->display("test grid");
        }
      }
    }

    // =======================================================
    // evaluate the expression graph, which the root node is A
    // treat operator as element wise
    //
    //    case 1: if A has fusion kernel, use it to evaluate and return
    //    case 2: if A is a data node, just return it's data 
    //    case 3: if A is not an element wise operator node 
    //            or need to update, evaluate it's child first,
    //            after that, evaluate the A
    // =======================================================
    ArrayPtr eval(NodePtr A) {
      // 1. Node has hash value, means may have a fusion kernel
      if (A->hash()) {
        // use A->hash() to get inside fusion kernel
        FusionKernelPtr fkptr = Jit_Driver::global()->get(A->hash());
        if (fkptr != NULL) {
          // prepare parameters used in fusion kernel
          vector<void*> list;
          vector<oa_int3> S;
          vector<ArrayPtr> update_list;
          PartitionPtr par_ptr;
          bitset<3> bt = A->get_bitset();
          vector<oa_int3> lb_list;
          vector<oa_int3> rb_list;
          oa_int3 lb_now = {{0,0,0}};
          oa_int3 rb_now = {{0,0,0}};
          vector<ArrayPtr> data_list;
          get_kernel_parameter_with_op(A, 
            list, update_list, S, par_ptr, bt,
            lb_list, rb_list, lb_now, rb_now,data_list);

          oa_int3 lb = A->get_lbound();
          oa_int3 rb = A->get_rbound();
          
          int sb = lb[0] + lb[1] + lb[2] + rb[0] + rb[1] + rb[2];
          int sz = update_list.size();
          vector<MPI_Request>  reqs_list;
          // pthread_t tid;
          // step 1:  start of update boundary
          int ranksize = oa::MPI::global()->size();
          if (sb && ranksize >1) {
            for (int i = 0; i < sz; i++){
              oa::funcs::update_ghost_start(update_list[i], reqs_list, 4, lb_list[i], rb_list[i]);
            }
            oa::funcs::update_ghost_end(reqs_list);
          }

          // put the answer array's data and shape into list
          ArrayPtr ap = ArrayPool::global()->get(par_ptr, A->get_data_type());
          S.push_back(ap->buffer_shape());
          S.push_back(A->get_lbound());
          S.push_back(A->get_rbound());
          S.push_back(ap->local_shape());

          list.push_back(ap->get_buffer());
          list.push_back((void*)S.data());
          void** list_pointer = list.data();
          
          // step 2:  calc_inside
          fkptr(list_pointer, ap->get_stencil_width());
          
          if (sb) {
            // step 3:  end of update boundary
              //oa::funcs::update_ghost_end(reqs_list);
            //oa::MPI::wait_end(&tid);

            // step 4:  calc_outside
            // use A->hash() + 1 to get outside fusion kernel
            //FusionKernelPtr out_fkptr = Jit_Driver::global()->get(A->hash() + 1);
            //if (out_fkptr) out_fkptr(list_pointer, ap->get_stencil_width());

            // set the boundary to zeros based on lb & rb becased it used illegal data
            oa::funcs::set_boundary_zeros(ap, lb, rb);
          }

          //cout<<"fusion-kernel called"<<endl;
          
          ap->set_bitset(A->get_bitset());
          ap->set_pos(A->get_pos());
          return ap;
        }
      }

      
      // 2. Node is a data node, just return the data
      if (A->has_data()) return A->get_data();

      // 3.1 Node is an operator node, and doesn't have fusion kernel
      // first, evaluate it's child node recursively
      vector<ArrayPtr> ops_ap;
      for (int i = 0; i < A->input_size(); i++) {
        ops_ap.push_back(eval(A->input(i)));
      }

      // 3.2 second, evaluate the node
      ArrayPtr ap;
      if(A->type() == TYPE_REF) {
        ap = oa::funcs::subarray(ops_ap[0], A->get_ref());
      } else {
        const NodeDesc& nd = get_node_desc(A->type());
        KernelPtr kernel_addr = nd.func;

        ap = kernel_addr(ops_ap);
        ap->set_bitset(A->get_bitset());
        ap->set_pos(A->get_pos());
      }
      return ap;
    }


    // =======================================================
    // Before evaluate the expression graph, we need to analyze the graph
    // Here's how we generate the fusion kernel for each sub expression graph
    // 
    // exp: A = AXF(A) + sub(A+B+C) 
    // there is two fusion kernels becasue of the sub operator
    //      kernel 1: ans = AXF(A) + tmp
    //      kernel 2: ans = A+B+C
    // =======================================================
    // NodePtr A :  the root of (sub)expression graph
    // is_root:     we only have to generate fusion kernels of the root node
    void gen_kernels_JIT_with_op(NodePtr A, bool is_root) {
      // 1. if A is a data node, doesn't have to generate fusion kernel
      if (A->has_data()) return ;
      
      const NodeDesc &nd = get_node_desc(A->type());
      
      // 2. if A is not element wise (like sum, rep, etc)
      //    need to generate it's children's fusion kernels recursively
      if (!nd.ew) {
        for (int i = 0; i < A->input_size(); i++) {
          gen_kernels_JIT_with_op(A->input(i), true);
        }
        return ;
      }

      // 3. if A's need update state is true, should generate fusion kernel 
      if (A->need_update()) {
        // should set update to false in order to generate kernels
        A->set_update(false);
        gen_kernels_JIT_with_op(A, true);
        A->set_update(true);
        return ;
      }

      // 4. is root && A->depth >= 2, generate fusion kernel
      if (is_root && A->get_depth() >= 1) {
        stringstream ss1;
        stringstream code;
        stringstream __code;
        stringstream __point;
				stringstream __code_const;

				stringstream __point_in;
				stringstream __point_out;
				//stringstream __code_const_in;
				//stringstream __code_const_out;

        // generate hash code for tree
        tree_to_string_stack(A, ss1);
        std::hash<string> str_hash;
        size_t hash = str_hash(ss1.str());
        if (g_debug) cout<<ss1.str()<<endl;
        if (g_debug) cout<<hash<<endl;
        
        // if already have kernel function ptr, do nothing
        if (Jit_Driver::global()->get(hash) != NULL) {
          if (g_debug) cout<<hash<<endl;
          A->set_hash(hash);
          // return ;   shouldn't return!!!!
        } 
        else {
          // else generate kernel function by JIT_Driver
          int id = 0;
          int S_id = 0;
          vector<int> int_id, float_id, double_id;
          tree_to_code_with_op(A, __code, __point, __point_in, __point_out, id, S_id, int_id, float_id, double_id);
          
          // JIT source code add function signature
          code_add_function_signature_with_op(code, hash);
          // JIT source code add const parameters
          code_add_const(__code_const, int_id, float_id, double_id);
          // JIT source code add calc_inside
          code_add_calc_inside(code, __code_const, __code, __point, __point_in, __point_out, int_id, float_id, double_id, A->get_data_type(), id, S_id, hash);

          // for debug
          if (g_debug) cout<<code.str()<<endl;

          // Add fusion kernel into JIT map
          Jit_Driver::global()->insert(hash, code);

          A->set_hash(hash);

          oa_int3 lb = A->get_lbound();
          oa_int3 rb = A->get_rbound();
          int sb = lb[0] + lb[1] + lb[2] + rb[0] + rb[1] + rb[2];

          // Add calc_outside
          if (sb) {
            //stringstream code_out;
            //size_t hash_out = hash + 1;
            //code_add_function_signature_with_op(code_out, hash_out);
            //code_add_const(code_out, int_id, float_id, double_id);
            //code_add_calc_outside(code_out, __code, A->get_data_type(), id, S_id);
            //// cout<<code_out.str()<<endl;
            //Jit_Driver::global()->insert(hash_out, code_out);
          }
        }
      }

      // 5. generate fusion kernels recursively 
      for (int i = 0; i < A->input_size(); i++) {
        gen_kernels_JIT_with_op(A->input(i), false);
      }
    }

    // example: (A1+S2)*A3
    void tree_to_string(NodePtr A, stringstream &ss) {
      const NodeDesc &nd = get_node_desc(A->type());
      
      // only data or non-element-wise
      if (A->has_data() || !nd.ew || A->need_update()) {
        if (A->is_seqs_scalar()) ss<<"S";
        else ss<<"A";
        ss<<A->get_data_type();
        return;
      }

      stringstream child[2];
      for (int i = 0; i < A->input_size(); i++) {
        tree_to_string(A->input(i), child[i]);
        //child[i] = tree_to_string(A->input(i));
      }

      switch(A->input_size()) {
      case 1:
        if(nd.sy == "abs")
          switch(A->get_data_type()) {
            case DATA_INT:
              ss<<"abs"<<"("<<child[0].str()<<")";
              break;
            case DATA_FLOAT:
              ss<<"fabsf"<<"("<<child[0].str()<<")";
              break;
            case DATA_DOUBLE:
              ss<<"fabs"<<"("<<child[0].str()<<")";
              break;    
            default:
              ss<<"fabs"<<"("<<child[0].str()<<")";
              break;    
          }
        else if(nd.sy == "sqrt")
          switch(A->get_data_type()) {
            case DATA_FLOAT:
              ss<<"sqrtf"<<"("<<child[0].str()<<")";
              break;
            default:
              ss<<"sqrt"<<"("<<child[0].str()<<")";
              break;    
          }
        else
          ss<<nd.sy<<"("<<child[0].str()<<")";
        break;
      case 2:
        if (nd.type == TYPE_POW) {
          ss<<"pow("<<child[0].str()<<","<<child[1].str()<<")";
        }
        else {
          ss<<"("<<child[0].str()<<")"<<nd.sy<<"("<<child[1].str()<<")";
        }
        break;
      }

      return;
    }

    void tree_to_code_with_op(NodePtr A, stringstream &ss, stringstream &__point, stringstream &__point_in, stringstream &__point_out, int &id, int& S_id,
      vector<int>& int_id, vector<int>& float_id, vector<int>& double_id) {
      const NodeDesc &nd = get_node_desc(A->type());

      // data node
      if (A->has_data() || !nd.ew || A->need_update()) {
        // scalar
        if (A->is_seqs_scalar()) {
          switch(A->get_data_type()) {
            case DATA_INT:
              ss<<"I_";
              ss<<int_id.size();
              int_id.push_back(id);
              break;
            case DATA_FLOAT:
              ss<<"F_";
              ss<<float_id.size();
              float_id.push_back(id);
              break;
            case DATA_DOUBLE:
              ss<<"D_";
              ss<<double_id.size();
              double_id.push_back(id);
              break;
          }
        // [i][j][k] based on node bitset
        } else {
          switch(A->get_data_type()) {
            case DATA_INT:
                __point<<"  int *list_"<<id<<";  list_"<<id<<" = (int *) list["<<id<<"];\n";

#ifdef __HAVE_CUDA__
								__point_in<<"list_"<<id<<", ";
								__point_out<<"int *list_"<<id<<", ";
#endif

              break;
            case DATA_FLOAT:
                __point<<"  float *list_"<<id<<";  list_"<<id<<" = (float *) list["<<id<<"];\n";

#ifdef __HAVE_CUDA__
								__point_in<<"list_"<<id<<", ";
								__point_out<<"float *list_"<<id<<", ";
#endif

              break;
            case DATA_DOUBLE:
                __point<<"  double *list_"<<id<<";  list_"<<id<<" = (double *) list["<<id<<"];\n";

#ifdef __HAVE_CUDA__
								__point_in<<"list_"<<id<<", ";
								__point_out<<"double *list_"<<id<<", ";
#endif

              break;    
          }
        /*
          ss<<"(";
          switch(A->get_data_type()) {
            case DATA_INT:
              ss<<"(int*)";
              break;
            case DATA_FLOAT:
              ss<<"(float*)";
              break;
            case DATA_DOUBLE:
              ss<<"(double*)";
              break;
          }
          */
          ss<<"list_"<<id;
          
          char pos_i[3] = "oi";
          char pos_j[3] = "oj";
          char pos_k[3] = "ok";
          
          bitset<3> bit = A->get_bitset();
          ss<<"[calc_id2("<<pos_i[bit[2]]<<",";
          ss<<pos_j[bit[1]]<<",";
          ss<<pos_k[bit[0]]<<",S"<<S_id<<"_0,S"<<S_id<<"_1)]";
          S_id++;
        }
        id++;
        return ;
      }

      stringstream child[2];
      for (int i = 0; i < A->input_size(); i++) {
        tree_to_code_with_op(A->input(i), child[i], __point, __point_in, __point_out, id, S_id, 
          int_id, float_id, double_id);
        //child[i] = tree_to_string(A->input(i));
      }

      switch(A->input_size()) {
      case 1:
        if (nd.type == TYPE_UNKNOWN) {
          string in = child[0].str();
          // printf("in Operator, k = %d\n", A->get_slice());
          string out  = replace_string(in, "k", to_string(A->get_slice()));
          ss<<out;
        }
        else change_string_with_op(ss, child[0].str(), nd);
        // bind grid if A.pos != -1
        if (A->get_pos() != -1) {
          if (nd.type == TYPE_DXC ||
            nd.type == TYPE_DYC ||
            nd.type == TYPE_DZC ||
            nd.type == TYPE_DXB ||
            nd.type == TYPE_DXF ||
            nd.type == TYPE_DYB ||
            nd.type == TYPE_DYF ||
            nd.type == TYPE_DZB ||
            nd.type == TYPE_DZF) {

            // get grid ptr
            ArrayPtr grid_ptr = Grid::global()->get_grid(A->get_pos(), nd.type);

            ss<<"/";
            switch(A->get_data_type()) {
              case DATA_INT:
                __point<<"  int *list_"<<id<<";  list_"<<id<<" = (int *) list["<<id<<"];\n";

#ifdef __HAVE_CUDA__
								__point_in<<"list_"<<id<<", ";
								__point_out<<"int *list_"<<id<<", ";
#endif

                break;
              case DATA_FLOAT:
                __point<<"  float *list_"<<id<<";  list_"<<id<<" = (float *) list["<<id<<"];\n";

#ifdef __HAVE_CUDA__
								__point_in<<"list_"<<id<<", ";
								__point_out<<"float *list_"<<id<<", ";
#endif

                break;
              case DATA_DOUBLE:
                __point<<"  double *list_"<<id<<";  list_"<<id<<" = (double *) list["<<id<<"];\n";

#ifdef __HAVE_CUDA__
								__point_in<<"list_"<<id<<", ";
								__point_out<<"double *list_"<<id<<", ";
#endif

                break;    
            }
            /*
            switch(grid_ptr->get_data_type()) {
              case DATA_INT:
                ss<<"(int*)";
                break;
              case DATA_FLOAT:
                ss<<"(float*)";
                break;
              case DATA_DOUBLE:
                ss<<"(double*)";
                break;
            }*/
            ss<<"list_"<<id;
            id++;
            
            char pos_i[3] = "oi";
            char pos_j[3] = "oj";
            char pos_k[3] = "ok";
            bitset<3> bit = grid_ptr->get_bitset();
            ss<<"[calc_id2("<<pos_i[bit[2]]<<",";
            ss<<pos_j[bit[1]]<<",";
            ss<<pos_k[bit[0]]<<",S"<<S_id<<"_0,S"<<S_id<<"_1)]";
            S_id++;
          }
        }
        break;
      case 2:
        if (nd.type == TYPE_POW) {
          ss<<"pow("<<child[0].str()<<","<<child[1].str()<<")";
        }
        else {
          ss<<"("<<child[0].str()<<")"<<nd.sy<<"("<<child[1].str()<<")";
        }
        break;
      }

      return;
    }

    void change_string_with_op(stringstream& ss, string in, const NodeDesc &nd) {
      string new_str1, new_str2, new_str;
      switch(nd.type) {
      // Central difference operator
      case TYPE_DXC:
        new_str1 = replace_string(in, "i,", "1+i,");
        new_str2 = replace_string(in, "i,", "-1+i,");
        ss<<"0.5*(("<<new_str1<<")-("<<new_str2<<"))";
        break;
      case TYPE_DYC:
        new_str1 = replace_string(in, "j,", "1+j,");
        new_str2 = replace_string(in, "j,", "-1+j,");
        ss<<"0.5*(("<<new_str1<<")-("<<new_str2<<"))";
        break;
      case TYPE_DZC:
        new_str1 = replace_string(in, "k,", "1+k,");
        new_str2 = replace_string(in, "k,", "-1+k,");
        ss<<"0.5*(("<<new_str1<<")-("<<new_str2<<"))";
        break;

      // average operator
      case TYPE_AXB:
        new_str = replace_string(in, "i,", "-1+i,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AXF:
        new_str = replace_string(in, "i,", "1+i,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AYB:
        new_str = replace_string(in, "j,", "-1+j,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AYF:
        new_str = replace_string(in, "j,", "1+j,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AZB:
        new_str = replace_string(in, "k,", "-1+k,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;
      case TYPE_AZF:
        new_str = replace_string(in, "k,", "1+k,");
        ss<<"0.5*(("<<in<<")+("<<new_str<<"))";
        break;

      // difference operator
      case TYPE_DXB:
        new_str = replace_string(in, "i,", "-1+i,");
        ss<<"1.0*(("<<in<<")-("<<new_str<<"))";
        break;
      case TYPE_DXF:
        new_str = replace_string(in, "i,", "1+i,");
        ss<<"1.0*(("<<new_str<<")-("<<in<<"))";
        break;
      case TYPE_DYB:
        new_str = replace_string(in, "j,", "-1+j,");
        ss<<"1.0*(("<<in<<")-("<<new_str<<"))";
        break;
      case TYPE_DYF:
        new_str = replace_string(in, "j,", "1+j,");
        ss<<"1.0*(("<<new_str<<")-("<<in<<"))";
        break;
      case TYPE_DZB:
        new_str = replace_string(in, "k,", "-1+k,");
        ss<<"1.0*(("<<in<<")-("<<new_str<<"))";
        break;
      case TYPE_DZF:
        new_str = replace_string(in, "k,", "1+k,");
        ss<<"1.0*(("<<new_str<<")-("<<in<<"))";
        break;

      // abs operator
      case TYPE_ABS:
        ss<<"fabs"<<"("<<in<<")";
        break;

      // other default monocular operator
      default:
        ss<<nd.sy<<"("<<in<<")";
        break;
      }
    }
    
    // replace all old_str in string in by new_str
    string replace_string(string& in, const string& old_str, const string& new_str) {
      string out = in;
      // use replace is not efficient, should be optimized later
      for(string::size_type i = 0; (i = out.find(old_str, i)) != string::npos;) {
        out.replace(i, old_str.length(), new_str);
        i += new_str.length();
      }
      return out;
    }



    void tree_to_string_stack(NodePtr A, stringstream &ss) {
      const NodeDesc &nd = get_node_desc(A->type());

      if (A->has_data() || !nd.ew || A->need_update()) {
        if (A->is_seqs_scalar()) ss<<"S";
        else {
          ss<<"A"<<A->get_bitset();
        }
        ss<<A->get_data_type();

        // if (A->need_update()) ss<<nd.sy;
        return ;
      }

      for (int i = 0; i < A->input_size(); i++) {
        tree_to_string_stack(A->input(i), ss);
      }
      ss<<nd.sy;

      return ;
    }

    void code_add_function_signature(stringstream& code, size_t& hash) {
      code<<"extern \"C\" {\nvoid kernel_"<<hash;
      code<<"(void** &list, int size) {\n";
    }

    void code_add_function_signature_with_op(stringstream& code, size_t& hash) {

      // code<<"#include <array>\n\n";
      // code<<"typedef std::array<int, 3> oa_int3;\n\n";
      code<<"#include \"math.h\"\n";
      code<<"#include \"stdlib.h\"\n";
      code<<"#include \"stdio.h\"\n";
      
      code<<"typedef int oa_int3[3];\n";
      code<<"#define min(a,b) ((a)<(b))?(a):(b)\n";
      code<<"#define BLOCK_NUM 32\n";
			code<<"#define calc_id2(i,j,k,S0,S1) ((k)*(S0)*(S1)+(j)*(S0)+(i))\n\n";

      code<<"extern \"C\" {\n";
      //code<<"#define calc_id2(i,j,k,S0,S1) ((k)*(S0)*(S1)+(j)*(S0)+(i))\n";
      //code<<"void kernel_"<<hash;
      //code<<"(void** &list, int o) {\n";
    }

    void code_add_const(stringstream& __code_const,
        vector<int>& int_id, vector<int>& float_id, vector<int>& double_id) {
      __code_const<<"\n";
      for (int i = 0; i < int_id.size(); i++) {
        __code_const<<"  const int I_"<<i<<" = ((int*)list["<<int_id[i]<<"])[0];\n";
      }
      for (int i = 0; i < float_id.size(); i++) {
        __code_const<<"  const float F_"<<i<<" = ((float*)list["<<float_id[i]<<"])[0];\n";
      }
      for (int i = 0; i < double_id.size(); i++) {
        __code_const<<"  const double D_"<<i<<" = ((double*)list["<<double_id[i]<<"])[0];\n";
      }
      __code_const<<"\n";
    }
    
    void code_add_function(stringstream& code, 
      stringstream& __code, DATA_TYPE dt, int& id) {

      code<<"  for (int i = 0; i < size; i++) {\n";  
      switch(dt) {
        case DATA_INT:
          code<<"    ((int*)(list["<<id<<"]))[i] = ";
          break;
        case DATA_FLOAT:
          code<<"    ((float*)(list["<<id<<"]))[i] = ";
          break;
        case DATA_DOUBLE:
          code<<"    ((double*)(list["<<id<<"]))[i] = ";
          break;    
      }
      code<<__code.str()<<";\n  }\n  return ;\n}}";
    }
/*
    void code_add_calc_outside(stringstream& code, 
      stringstream& __code, DATA_TYPE dt, int& id, int& S_id) {
      
      code<<"  oa_int3* oa_int3_p = (oa_int3*)(list["<<id + 1<<"]);\n";
      for (int i = 0; i <= S_id; i++) {
        code<<"  const oa_int3 &S"<<i<<" = oa_int3_p["<<i<<"];\n";
        code<<"  const int S"<<i<<"_0 = oa_int3_p["<<i<<"][0];\n";
        code<<"  const int S"<<i<<"_1 = oa_int3_p["<<i<<"][1];\n";
      }
      code<<"\n";
      code<<"  const oa_int3 &lbound = oa_int3_p["<<S_id + 1<<"];\n";
      code<<"  const oa_int3 &rbound = oa_int3_p["<<S_id + 2<<"];\n";
      code<<"  const oa_int3 &sp = oa_int3_p["<<S_id + 3<<"];\n\n";

      string ans_type[3];
      ans_type[DATA_INT] = "(int*)";
      ans_type[DATA_FLOAT] = "(float*)";
      ans_type[DATA_DOUBLE] = "(double*)";

      // lbound[2]
      code<<"  if (lbound[2]) {\n";
      code<<"    for (int k = o; k < o + lbound[2]; k++) {\n";
      code<<"      for (int j = o; j < o + sp[1]; j++) {\n";
      code<<"      #pragma simd\n";
      code<<"      #pragma clang loop vectorize(assume_safety)\n";
      code<<"      #pragma clang loop interleave(enable)\n";
      code<<"      #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"        for (int i = o; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";
      
      // rbound[2]
      code<<"  if (rbound[2]) {\n";
      code<<"    for (int k = o + sp[2] - rbound[2]; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o; j < o + sp[1]; j++) {\n";
      code<<"      #pragma simd\n";
      code<<"      #pragma clang loop vectorize(assume_safety)\n";
      code<<"      #pragma clang loop interleave(enable)\n";
      code<<"      #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"        for (int i = o; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";


      // lbound[1]
      code<<"  if (lbound[1]) {\n";
      code<<"    for (int k = o; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o; j < o + lbound[1]; j++) {\n";
      code<<"      #pragma simd\n";
      code<<"      #pragma clang loop vectorize(assume_safety)\n";
      code<<"      #pragma clang loop interleave(enable)\n";
      code<<"      #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"        for (int i = o; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";
      
      // rbound[1]
      code<<"  if (rbound[1]) {\n";
      code<<"    for (int k = o; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o + sp[1] - rbound[1]; j < o + sp[1]; j++) {\n";
      code<<"      #pragma simd\n";
      code<<"      #pragma clang loop vectorize(assume_safety)\n";
      code<<"      #pragma clang loop interleave(enable)\n";
      code<<"      #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"        for (int i = o; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";

      // lbound[0]
      code<<"  if (lbound[0]) {\n";
      code<<"    for (int k = o; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o; j < o + sp[1]; j++) {\n";
      code<<"        for (int i = o; i < o + lbound[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";
      
      // rbound[0]
      code<<"  if (rbound[0]) {\n";
      code<<"    for (int k = o; k < o + sp[2]; k++) {\n";
      code<<"      for (int j = o; j < o + sp[1]; j++) {\n";
      code<<"        for (int i = o + sp[0] - rbound[0]; i < o + sp[0]; i++) {\n";
      code<<"          ("<<ans_type[dt]<<"(list["<<id<<"]))[calc_id2(i,j,k,S"
                       <<S_id<<"_0,S"<<S_id<<"_1)] = "<<__code.str()<<";\n";
      code<<"        }\n";
      code<<"      }\n";
      code<<"    }\n";
      code<<"  }\n\n";

      code<<"  return ;\n}}";

    }
    */

     void code_add_calc_inside(stringstream& code, stringstream& __code_const,
      stringstream& __code, stringstream& __point, stringstream& __point_in,  stringstream& __point_out, 
			vector<int>& int_id, vector<int>& float_id, vector<int>& double_id, 
			DATA_TYPE dt, int& id, int& S_id, size_t& hash) {
// ************* GPU kernel begin ************* //
#ifdef __HAVE_CUDA__
      switch(dt) {
        case DATA_INT:
						__point_in<<"list_"<<id<<", ";
						__point_out<<"int *list_"<<id<<", ";
          break;
        case DATA_FLOAT:
						__point_in<<"list_"<<id<<", ";
						__point_out<<"float *list_"<<id<<", ";
          break;
        case DATA_DOUBLE:
						__point_in<<"list_"<<id<<", ";
						__point_out<<"double *list_"<<id<<", ";
          break;    
      }

			code<<"__global__ void kernel_gpu_"<<hash<<"(";

      for (int i = 0; i < int_id.size(); i++) {
				code<<"int* list_"<<int_id[i]<<", ";
      }
      for (int i = 0; i < float_id.size(); i++) {
				code<<"float* list_"<<float_id[i]<<", ";
      }
      for (int i = 0; i < double_id.size(); i++) {
				code<<"double* list_"<<double_id[i]<<", ";
      }

			code<<__point_out.str();
			code<<"int ied, int jed, int ked, ";
		
     for (int i = 0; i <= S_id; i++) {
        //code<<"  const oa_int3 &S"<<i<<" = oa_int3_p["<<i<<"];\n";
        code<<"int S"<<i<<"_0, ";
        code<<"int S"<<i<<"_1, ";
      }

      code<<"int o) {\n";

			code<<"  int i = threadIdx.x + blockIdx.x * blockDim.x + o;\n";
			code<<"  int j = threadIdx.y + blockIdx.y * blockDim.y + o;\n";
			code<<"  int k = threadIdx.z + blockIdx.z * blockDim.z + o;\n\n";

			code<<"  if(i >= ied || j >= jed || k >= ked)  return;\n";
			code<<"  else {\n";

      for (int i = 0; i < int_id.size(); i++) {
				code<<"    const int I_"<<i<<" = list_"<<int_id[i]<<"[0];\n";
      }
      for (int i = 0; i < float_id.size(); i++) {
				code<<"    const float F_"<<i<<" = list_"<<float_id[i]<<"[0];\n";
      }
      for (int i = 0; i < double_id.size(); i++) {
				code<<"    const double D_"<<i<<" = list_"<<double_id[i]<<"[0];\n";
      }

	    code<<"	   	 list_"<<id<<"[calc_id2(i,j,k,S"<<S_id<<"_0,S"<<S_id<<"_1)] = ";
			code<<__code.str()<<";\n  }\n\n";
			code<<"  //__syncthreads();\n}\n\n";
#endif
// ************* GPU kernel end ************* //

// ************* C++ kernel begin ************* //
 		  code<<"void kernel_"<<hash;
 			code<<"(void** &list, int o) {\n";
			code<<"  //o = 1;//temp wangdong\n";
      code<<"  oa_int3* oa_int3_p = (oa_int3*)(list["<<id + 1<<"]);\n\n";

#ifndef __HAVE_CUDA__
      code<<__code_const.str();
#else
      for (int i = 0; i < int_id.size(); i++) {
      	code<<"  int *list_"<<int_id[i]<<";  list_"<<int_id[i]<<" = (int *) list["<<int_id[i]<<"];\n";
      }
      for (int i = 0; i < float_id.size(); i++) {
      	code<<"  float *list_"<<float_id[i]<<";  list_"<<float_id[i]<<" = (float *) list["<<float_id[i]<<"];\n";
      }
      for (int i = 0; i < double_id.size(); i++) {
      	code<<"  double *list_"<<double_id[i]<<";  list_"<<double_id[i]<<" = (double *) list["<<double_id[i]<<"];\n";
      }
#endif

      code<<__point.str();

      switch(dt) {
        case DATA_INT:
            code<<"  int *list_"<<id<<";  list_"<<id<<" = (int *) list["<<id<<"];\n";
          break;
        case DATA_FLOAT:
            code<<"  float *list_"<<id<<";  list_"<<id<<" = (float *) list["<<id<<"];\n";
          break;
        case DATA_DOUBLE:
            code<<"  double *list_"<<id<<";  list_"<<id<<" = (double *) list["<<id<<"];\n";
          break;    
      }

      code<<"\n";
      code<<"  const oa_int3 &lbound = oa_int3_p["<<S_id + 1<<"];\n";
      code<<"  const oa_int3 &rbound = oa_int3_p["<<S_id + 2<<"];\n";
      code<<"  const oa_int3 &sp = oa_int3_p["<<S_id + 3<<"];\n\n";

      for (int i = 0; i <= S_id; i++) {
        code<<"  const int S"<<i<<"_0 = oa_int3_p["<<i<<"][0];  ";
        code<<"  const int S"<<i<<"_1 = oa_int3_p["<<i<<"][1];\n";
      }

      code<<"  int ist=o ; ";
      code<<"  int ied=o + sp[0] ;\n";
      code<<"  int jst=o ; ";
      code<<"  int jed=o + sp[1] ;\n";
      code<<"  int kst=o ; ";
      code<<"  int ked=o + sp[2] ;\n\n";

#ifdef __HAVE_CUDA__
			//code<<"  const int N = sp[0] * sp[1] *sp[2];\n";
			code<<"  dim3 threadPerBlock(8,8,8);\n";
			code<<"  dim3 num_blocks((sp[0]+7)/8, (sp[1]+7)/8, (sp[2]+7)/8);\n";
			//code<<"  for(int k = kst; k < ked; k++)\n";
			code<<"  kernel_gpu_"<<hash<<"<<<num_blocks,threadPerBlock>>>(";
      for (int i = 0; i < int_id.size(); i++) {
      	code<<"list_"<<int_id[i]<<", ";        
			}
      for (int i = 0; i < float_id.size(); i++) {
      	code<<"list_"<<float_id[i]<<", ";        
      }
      for (int i = 0; i < double_id.size(); i++) {
      	code<<"list_"<<double_id[i]<<", ";        
      }
			code<<__point_in.str();
			code<<"ied, jed, ked, ";
      for (int i = 0; i <= S_id; i++) {
        code<<"S"<<i<<"_0, ";
        code<<"S"<<i<<"_1, ";
      }
			code<<"o);\n";
#else
      code<<"  /*for (int kk = kst; kk< ked+BLOCK_NUM; kk += BLOCK_NUM)*/{\n";
      code<<"    //int kend=min(kk+BLOCK_NUM,ked);\n";
      code<<"    /*for (int jj = jst; jj< jed+BLOCK_NUM; jj += BLOCK_NUM)*/{\n";
      code<<"      //int jend=min(jj+BLOCK_NUM,jed);\n";
      code<<"      /*for (int ii = ist; ii< ied+BLOCK_NUM; ii += BLOCK_NUM)*/{\n";
      code<<"        //int iend=min(ii+BLOCK_NUM,ied);\n";
      code<<"        for (int k = kst; k < ked; k++) {\n";
      code<<"          for (int j = jst; j < jed; j++) {\n";
      code<<"            #pragma simd\n";
      code<<"            #pragma clang loop vectorize(assume_safety)\n";
      code<<"            #pragma clang loop interleave(enable)\n";
      code<<"            #pragma clang loop vectorize_width(8) interleave_count(1)\n";
      code<<"            for (int i = ist; i < ied ;i++){\n";

      code<<"              list_"<<id<<"[calc_id2(i,j,k,S"<<S_id<<"_0,S"<<S_id<<"_1)] = ";
      code<<__code.str();
			code<<";\n            }\n          }\n        }\n      }\n    }\n  }\n";
#endif

      // code<<__code.str()<<";\n   printf(\"##:%d %d %d\\n\", o, o + lbound[2], o + sp[2] - rbound[2]);  \n  }\n    }\n  }\n  return ;\n}}";

      code<<"  return ;\n}}\n\n";
// ************* C++ kernel end ************* //
    }  

  }
}
