
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Kernel_Functions.cpp

#define __CLASS__ "Kernel_Functions::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <string.h>

// ===== Import/Includes ====================================================
#include "KmsLib/Exception.h"

// ===== OpenNet ============================================================
#include "Kernel_Functions.h"

// Constants
/////////////////////////////////////////////////////////////////////////////

#define EOL "\n"

static const char * CODE_KERNEL_C =
"    }" EOL
"}"     EOL;

static const char * CODE_FILE_HEADER =
"#include <OpenNetK/Kernel.h>" EOL;

static const char * CODE_KERNEL_A =
"OPEN_NET_KERNEL void Filter(" EOL;

#ifdef _KMS_LINUX_

    static const char * CODE_KERNEL_B =
    " )"                        EOL
    "{"                         EOL
    "    switch ( blockIdx.x )" EOL
    "    {"                     EOL;

#endif

#ifdef _KMS_WINDOWS_

    static const char * CODE_KERNEL_B =
    " )"                               EOL
    "{"                                EOL
    "    switch ( get_group_id( 0 ) )" EOL
    "    {"                            EOL;

#endif

// Public
/////////////////////////////////////////////////////////////////////////////

// Exception  KmsLib::Exception *  CODE_ERROR
Kernel_Functions::Kernel_Functions()
{
    OpenNet::Status lStatus = SetCode(CODE_FILE_HEADER, static_cast<unsigned int>(strlen(CODE_FILE_HEADER)), 1 );
    if (OpenNet::STATUS_OK != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
            "SourceCode::SetCode( ,  ) failed", NULL, __FILE__, __CLASS__ "Kernel_Functions", __LINE__, lStatus);
    }
}

Kernel_Functions::~Kernel_Functions()
{
}

// aBufferQty [---;R--]
//
// Exception  KmsLib::Exception *  See Kernel_Functions::AppendCode
void Kernel_Functions::AddDispatchCode(const unsigned int * aBufferQty)
{
    assert(NULL != aBufferQty);

    assert(0 < mFunctionNames.size());

    AppendCode(CODE_KERNEL_A);

    char lCode[64];

    unsigned int lFunctionCount = static_cast< unsigned int >( mFunctionNames.size() );
    unsigned int i;
    unsigned int j;

    for (i = 0; i < lFunctionCount; i++)
    {
        assert(0 < aBufferQty[i]);

        for (j = 0; j < aBufferQty[i]; j++)
        {
            #ifdef _KMS_LINUX_
                sprintf_s(lCode, "    OpenNet_BufferHeader * aBH_%u_%u", i, j);
            #endif

            #ifdef _KMS_WINDOWS_
                sprintf_s(lCode, "    __global OpenNet_BufferHeader * aBH_%u_%u", i, j);
            #endif

            AppendCode(lCode);

            if ((lFunctionCount > (i + 1)) || (aBufferQty[i] > (j + 1)))
            {
                AppendCode("," EOL);
            }
        }
    }

    AppendCode(CODE_KERNEL_B);

    unsigned int lId = 0;

    for (i = 0; i < lFunctionCount; i++)
    {
        for (j = 0; j < aBufferQty[i]; j++)
        {
            sprintf_s(lCode, "    case %2u : %s( aBH_%u_%u ); break;" EOL, lId, mFunctionNames[ i ].c_str(), i, j);
            AppendCode(lCode);
            lId++;
        }
    }

    AppendCode(CODE_KERNEL_C);
}

// aFunction [---;R--]
//
// Exception  KmsLib::Exception *  CODE_ERROR
void Kernel_Functions::AddFunction(const OpenNet::Function & aFunction)
{
    OpenNet::Status lStatus = SourceCode::AppendCode(aFunction);
    if (OpenNet::STATUS_OK != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
            "OpenNet::SourceCode::AppenCode(  ) failed", NULL, __FILE__, __CLASS__ "AddFunction", __LINE__, lStatus);
    }

    mFunctionNames.push_back(aFunction.GetFunctionName());
}

// Private
/////////////////////////////////////////////////////////////////////////////

// aCode [---;R--]
//
// Exception  KmsLib::Exception *  CODE_ERROR
void Kernel_Functions::AppendCode(const char * aCode)
{
    assert(NULL != aCode);

    OpenNet::Status lStatus = SourceCode::AppendCode(aCode, static_cast<unsigned int>(strlen(aCode)));
    if (OpenNet::STATUS_OK != lStatus)
    {
        throw new KmsLib::Exception(KmsLib::Exception::CODE_ERROR,
            "SourceCode::AppendCode( ,  ) failed", NULL, __FILE__, __CLASS__ "AppendCode", __LINE__, lStatus);
    }
}
