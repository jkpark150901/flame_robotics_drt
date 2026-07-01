/**
 * @file basic.interface.hpp
 * @author your name (you@domain.com)
 * @brief 
 * @version 0.1
 * @date 2026-05-19
 * 
 * @copyright Copyright (c) 2026
 * 
 */

#ifndef FLAME_BASIC_INTERFACE_HPP_INCLUDED
#define FLAME_BASIC_INTERFACE_HPP_INCLUDED

#include <flame/component/object.hpp>
#include <atomic>
#include <thread>


using namespace std;
using namespace flame::component;

class basic_interface : public flame::component::Object {
    public:
        basic_interface();
        virtual ~basic_interface() = default;

        /* default interface functions */
        bool onInit() override;
        void onLoop() override;
        void onClose() override;
        void onData(flame::component::ZData& data) override;
};

EXPORT_COMPONENT_API

#endif