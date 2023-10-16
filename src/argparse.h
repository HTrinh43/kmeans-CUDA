#pragma once
#ifndef _ARGPARSE_H
#define _ARGPARSE_H

#include <getopt.h>
#include <stdlib.h>
#include <iostream>

struct options_t {
    char* in_file;
    int k;
    int d;
    int m;
    double t;
    bool c;
    int s;
    int r;
};

void get_opts(int argc, char** argv, struct options_t* opts);
#endif
