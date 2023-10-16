#include <argparse.h>

void get_opts(int argc,
    char** argv,
    struct options_t* opts)
{
    if (argc == 1)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "\t-k num_cluster" << std::endl;
        std::cout << "\t-d dims" << std::endl;
        std::cout << "\t-i inputfilename" << std::endl;
        std::cout << "\t-m max_num_iter" << std::endl;
        std::cout << "\t-t threshold" << std::endl;
        std::cout << "\t-c" << std::endl;
        std::cout << "\t-s seed" << std::endl;
        std::cout << "\t-r running_method (s: sequential, c: CUDA shared memory, t: Thrust)" << std::endl;
        exit(0);
    }


    struct option l_opts[] = {
        {"k",required_argument, NULL, 'k'},
        {"d",required_argument, NULL, 'd'},
        {"i",required_argument, NULL, 'i'},
        {"m",required_argument, NULL, 'm'},
        {"t",required_argument, NULL, 't'},
        {"c",no_argument, NULL, 'c'},
        {"s",no_argument, NULL, 's'},
        {"r",no_argument, NULL, 'r'}
    };
    opts->r = 0;
    opts->s = 1;
    opts->c = false;
    int ind, c;
    while ((c = getopt_long(argc, argv, "k:d:i:m:t:cs:r:", l_opts, &ind)) != -1)
    {
        switch (c)
        {
        case 0:
            break;
        case 'k':
            opts->k = atoi((char*)optarg);
            break;
        case 'd':
            opts->d = atoi((char*)optarg);
            break;
        case 'i':
            opts->in_file = (char*)optarg;
            break;
        case 'm':
            opts->m = atoi((char*)optarg);
            break;
        case 't':
            char *endptr;
            opts->t = strtod((char*)optarg, &endptr);
            break;
        case 'c':
            opts->c = true;
            break;
        case 's':
            opts->s = atoi((char*)optarg);
            break;
        case 'r':
            opts->r = atoi((char*)optarg);
            break;
        case ':':
            std::cerr << argv[0] << ": option -" << (char)optopt << "requires an argument." << std::endl;
            exit(1);
        }
    }
}
