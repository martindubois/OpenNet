
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Utils.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== OpenNet ============================================================
#include "Utils.h"

// Functions
/////////////////////////////////////////////////////////////////////////////

OpenNet::Status Utl_ExceptionToStatus(KmsLib::Exception * aE)
{
    assert(NULL != aE);

    switch (aE->GetCode())
    {
    case KmsLib::Exception::CODE_IOCTL_ERROR      : return OpenNet::STATUS_IOCTL_ERROR    ;
    case KmsLib::Exception::CODE_NOT_ENOUGH_MEMORY: return OpenNet::STATUS_TOO_MANY_BUFFER;
    case KmsLib::Exception::CODE_OPEN_CL_ERROR    : return OpenNet::STATUS_OPEN_CL_ERROR  ;
    }

    printf("%s ==> STATUS_EXCEPTION\n", KmsLib::Exception::GetCodeName(aE->GetCode()));
    aE->Write(stdout);

    return OpenNet::STATUS_EXCEPTION;
}
