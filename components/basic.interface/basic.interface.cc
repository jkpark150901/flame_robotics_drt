#include "basic.interface.hpp"
#include <flame/log.hpp>

/* create component instance */
static basic_interface* _instance = nullptr;
flame::component::Object* Create(){ if(!_instance) _instance = new basic_interface(); return _instance; }
void Release(){ if(_instance){ delete _instance; _instance = nullptr; }}

basic_interface::basic_interface() {
}

bool basic_interface::onInit(){
    try{
        json parameters = getProfile()->parameters();
    }
    catch(json::exception& e){
        logger::error("[{}] Profile Error : {}", getName(), e.what());
        return false;
    }
    catch(const std::exception& e){
        logger::error("[{}] Initialization Error : {}", getName(), e.what());
        return false;
    }

    return true;
}

void basic_interface::onLoop(){
}

void basic_interface::onClose(){
    

}

void basic_interface::onData(flame::component::ZData& data){
}
