#pragma once

#include <vector>
#include <string>
#include <sstream>

// this is garbage
namespace local
{
    namespace detail
    {
        static bool ends_with(const std::string& a, const std::string& b)
        {
            if (b.size() > a.size()) return false;
            return std::equal(a.begin() + a.size() - b.size(), a.end(), b.begin());
        }
    }
    struct io_control_t
    {
        std::vector<std::string> raw_args;
        std::string input_filename;
        int init_num;
        bool init_from_file;
        io_control_t(int argc, char** argv)
        {
            init_num = -1;
            init_from_file = false;
            for (int i = 0; i < argc; ++i)
            {
                raw_args.push_back(std::string(argv[i]));
            }
            
            input_filename = "input.ptl";
            for (const auto & s: raw_args)
            {
                if (detail::ends_with(s, ".ptl")) input_filename = s;
            }
            spade::cli_args::shortname_args_t args(argc, argv);
            if (args.has_arg("-init"))
            {
                init_from_file = true;
                std::string init_num_str = args["-init"];
                std::istringstream iss(init_num_str);
                iss >> init_num;
                if (iss.fail())
                {
                    print("BAD INIT INTEGER");
                    exit(1);
                }
            }
        }
        
        bool is_init() const {return init_from_file;}
        int get_init_num() const {return init_num;}
        
        std::string get_input_file_name() const {return input_filename;}
        
        std::string ck_dir_name(const int i)  const {return std::string("ck") + std::to_string(i);}
        std::string ck_file_name(const int v, const int i) const {return ck_dir_name(v) + "/ck_" + spade::utils::zfill(i, 8) + ".bin";}
        
        std::string sl_dir_name(const int i)  const {return std::string("sl") + std::to_string(i);}
        std::string sl_file_name(const int i) const {return "sl_" + spade::utils::zfill(i, 8);}
        
        void create_if_not_exist(const std::string& dir_name) const
        {
            std::filesystem::path out_path(dir_name);
            if (!std::filesystem::is_directory(out_path)) std::filesystem::create_directory(out_path);
        }
        
        void create_dirs() const
        {
            create_if_not_exist(ck_dir_name(0));
            create_if_not_exist(ck_dir_name(1));
        }
    };
}