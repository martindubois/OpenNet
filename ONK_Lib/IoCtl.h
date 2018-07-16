
// Author   KMS - Martin Dubois, ing
// Product  OpenNet
// File     ONK_Lib/IoCtl.h

#pragma once

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    unsigned int mIn_MinSize_byte ;
    unsigned int mOut_MinSize_byte;
}
IoCtl_Info;

typedef enum
{
    IOCTL_RESULT_OK                = 0x00000000,

    IOCTL_RESULT_PROCESSING_NEEDED = 0xffffffe0,

    IOCTL_RESULT_ERROR             = 0xfffffff9,
    IOCTL_RESULT_INVALID_PARAMETER = 0xfffffffa,
    IOCTL_RESULT_INVALID_SYSTEM_ID = 0xfffffffb,
    IOCTL_RESULT_NO_BUFFER         = 0xfffffffc,
    IOCTL_RESULT_NOT_SET           = 0xfffffffd,
    IOCTL_RESULT_TOO_MANY_ADAPTER  = 0xfffffffe,
    IOCTL_RESULT_TOO_MANY_BUFFER   = 0xffffffff,
}
IoCtl_Result;
