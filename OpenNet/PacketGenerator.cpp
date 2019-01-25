
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/PacketGenerator.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>

// ===== OpenNet ============================================================
#include "PacketGenerator_Internal.h"

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             PacketGenerator_Internal contructor raise an exception
    PacketGenerator * PacketGenerator::Create()
    {
        PacketGenerator * lResult;

        try
        {
            lResult = new PacketGenerator_Internal();
        }
        catch (...)
        {
            lResult = NULL;
        }

        return lResult;
    }

    // NOT TESTED  OpenNet.System.ErrorHandling
    //             System_Internal destructor raise an exception
    void PacketGenerator::Delete()
    {
        try
        {
            delete this;
        }
        catch (...)
        {
        }
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    PacketGenerator::PacketGenerator()
    {
    }

    PacketGenerator::~PacketGenerator()
    {
    }

}
