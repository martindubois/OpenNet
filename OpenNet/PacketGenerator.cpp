
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/PacketGenerator.cpp

#define __CLASS__ "PacketGenerator::"

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
            // new ==> delete  See Delete
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
            // printf( __CLASS__ "~Delete - delete 0x%lx (this)\n", reinterpret_cast< uint64_t >( this ) );

            // new ==> delete  See Create
            delete this;
        }
        catch (...)
        {
            printf( __CLASS__ "~Delete - Exception\n" );
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
