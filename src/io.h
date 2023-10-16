#ifndef _IO_H
#define _IO_H

#include <argparse.h>
#include <iostream>
#include <fstream>
#include <helpers.h>

void read_file(struct options_t* args,
               int*              n_vals,
               std::vector<Point>& dataSet,
               Point_cu& dataSet2);

#endif
