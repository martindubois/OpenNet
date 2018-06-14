
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Processor_Internal.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Processor.h>

// Class
/////////////////////////////////////////////////////////////////////////////

class Processor_Internal : public OpenNet::Processor
{

public:

    Processor_Internal(cl_platform_id aPlatform, cl_device_id aDevice);

    // ===== OpenNet::Processor =============================================
    virtual OpenNet::Status GetInfo(Info * aOut) const;
    virtual OpenNet::Status Display(FILE * aOut) const;

private:

    void InitInfo();

    bool GetDeviceInfo(cl_device_info aParam);
    void GetDeviceInfo(cl_device_info aParam, size_t aOutSize_byte, void * aOut);

    cl_context       mContext;
    cl_device_id     mDevice ;
    Info             mInfo   ;
    cl_command_queue mQueue  ;

};
