
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) KMS 2018-2019. All rights reserved.
// Product    OpenNet
// File       OpenNet/SourceCode.cpp

#define __CLASS__ "SourceCode::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <stdint.h>
#include <string.h>

#ifdef _KMS_WINDOWS_
    // ===== Windows ========================================================
    #include <Windows.h>
#endif

// ===== Includes/OpenNet ===================================================
#include <OpenNet/SourceCode.h>

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static char * Allocate(unsigned int aSize_byte);

#ifdef _KMS_WINDOWS_
    static OpenNet::Status GetInputFileSize(HANDLE aHandle, unsigned int * aOut_byte);
    static OpenNet::Status OpenInputFile   (const char * aFileName, HANDLE * aOut);
    static OpenNet::Status ReadInputFile   (HANDLE aHandle, void * aOut, unsigned int aOutSize_byte);
#endif

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    SourceCode::SourceCode() : mArgumentCount( 1 ), mCode(NULL), mCodeSize_byte(0)
    {
        memset(&mName, 0, sizeof(mName));
    }

    SourceCode::~SourceCode()
    {
        if (NULL != mCode)
        {
            // printf( __CLASS__ "~SourceCode - delete [] 0x%lx (mCode)\n", reinterpret_cast< uint64_t >( mCode ) );

            delete [] mCode;
        }
    }

    Status SourceCode::AppendCode(const char * aCode, unsigned int aCodeSize_byte)
    {
        if (NULL == aCode)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        if (0 >= aCodeSize_byte)
        {
            return STATUS_EMPTY_CODE;
        }

        char * lNewCode = Allocate(mCodeSize_byte + aCodeSize_byte + 1);
        assert(NULL != lNewCode);

        memcpy(lNewCode                 , mCode, mCodeSize_byte);
        memcpy(lNewCode + mCodeSize_byte, aCode, aCodeSize_byte);

        // printf( __CLASS__ "AppendCode - delete [] 0x%lx (mCode)\n", reinterpret_cast< uint64_t >( mCode ) );

        delete[] mCode;

        mCode           = lNewCode      ;
        mCodeSize_byte += aCodeSize_byte;

        return STATUS_OK;
    }

    Status SourceCode::AppendCode(const SourceCode & aCode)
    {
        if (NULL == (&aCode))
        {
            return STATUS_INVALID_REFERENCE;
        }

        return AppendCode(aCode.GetCode(), aCode.GetCodeSize());
    }

    unsigned int SourceCode::GetCodeSize() const
    {
        if (NULL == mCode)
        {
            return 0;
        }

        assert(0 < mCodeSize_byte);

        return mCodeSize_byte;
    }

    const char * SourceCode::GetName() const
    {
        return mName;
    }

    Status SourceCode::ResetCode()
    {
        if (NULL == mCode)
        {
            return STATUS_CODE_NOT_SET;
        }

        ReleaseCode();

        return STATUS_OK;
    }

    Status SourceCode::SetArgumentCount(unsigned int aArgCount)
    {
        assert(1 <= mArgumentCount);

        if (1 > aArgCount)
        {
            return STATUS_INVALID_ARGUMENT_COUNT;
        }

        mArgumentCount = aArgCount;

        return STATUS_OK;
    }

    // NOT TESTED  OpenNet.Filter.ErrorHandling
    //             CloseHandle fail<br>
    //             ReadInputFile fail
    Status SourceCode::SetCode(const char * aFileName)
    {
        if (NULL != mCode)
        {
            return STATUS_CODE_ALREADY_SET;
        }

        assert(0 == mCodeSize_byte);

        #ifdef _KMS_LINUX_

            return STATUS_NOT_IMPLEMENTED;

        #endif

        #ifdef _KMS_WINDOWS_

            HANDLE lHandle;

            Status lResult = OpenInputFile(aFileName, &lHandle);
            if (STATUS_OK != lResult)
            {
                return lResult;
            }

            lResult = GetInputFileSize(lHandle, &mCodeSize_byte);
            if (STATUS_OK == lResult)
            {
                mCode = Allocate(mCodeSize_byte);
                assert(NULL != mCode);

                lResult = ReadInputFile(lHandle, mCode, mCodeSize_byte);
                if (STATUS_OK != lResult)
                {
                    ReleaseCode();
                    assert(NULL == mCode         );
                    assert(   0 == mCodeSize_byte);
                }
            }

            if (!CloseHandle(lHandle))
            {
                lResult = STATUS_ERROR_CLOSING_FILE;
            }

            return lResult;

        #endif
    }

    Status SourceCode::SetCode(const char * aCode, unsigned int aSize_byte)
    {
        if (NULL == aCode)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        if (0 >= aSize_byte)
        {
            return STATUS_EMPTY_CODE;
        }

        if (NULL != mCode)
        {
            return STATUS_CODE_ALREADY_SET;
        }

        assert( 0 == mCodeSize_byte );

        mCode = Allocate(aSize_byte + 1);
        assert(NULL != mCode);

        memcpy(mCode, aCode, aSize_byte + 1);

        mCodeSize_byte = aSize_byte;

        return STATUS_OK;
    }

    Status SourceCode::SetName(const char * aName)
    {
        if (NULL == aName)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        strncpy_s(mName, aName, sizeof(mName) - 1);

        return STATUS_OK;
    }

    Status SourceCode::Display(FILE * aOut) const
    {
        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "    SourceCode :\n");

        if (NULL == mCode)
        {
            fprintf(aOut, "      Code not set\n");
        }

        fprintf(aOut, "      Arg. Count = %u\n"      , mArgumentCount);
        fprintf(aOut, "      Code Size  = %u bytes\n", mCodeSize_byte);
        fprintf(aOut, "      Name       = %s\n"      , mName         );

        if (NULL != mCode)
        {
            fprintf(aOut, "      Code      =\n");
            fprintf(aOut, "%s\n", mCode);
        }

        return STATUS_OK;
    }

    unsigned int SourceCode::Edit_Remove(const char * aSearch)
    {
        if (NULL == aSearch)
        {
            return 0;
        }

        if (NULL == mCode)
        {
            return 0;
        }

        unsigned int lSearchLength = static_cast<unsigned int>(strlen(aSearch));

        if (0 >= lSearchLength)
        {
            return 0;
        }

        char       * lPtr    = mCode;
        unsigned int lResult = 0;

        while (NULL != (lPtr = strstr(lPtr, aSearch)))
        {
            char * lSrc = lPtr + lSearchLength;
            memmove(lPtr, lSrc, strlen(lSrc) + 1);
            lResult++;
        }

        mCodeSize_byte -= lResult * lSearchLength;

        return lResult;
    }

    unsigned int SourceCode::Edit_Replace(const char * aSearch, const char * aReplace)
    {
        if (NULL == aSearch)
        {
            return 0;
        }

        if (NULL == aReplace)
        {
            return Edit_Remove(aSearch);
        }

        if (NULL == mCode)
        {
            return 0;
        }

        unsigned int lSearchLength  = static_cast<unsigned int>(strlen(aSearch ));
        unsigned int lReplaceLength = static_cast<unsigned int>(strlen(aReplace));

        if (0 >= lSearchLength)
        {
            return 0;
        }

        unsigned int lResult;

        if      (            0 == lReplaceLength) { lResult = Edit_Remove           (aSearch); }
        else if (lSearchLength == lReplaceLength) { lResult = Edit_Replace_ByEqual  (aSearch, aReplace, lSearchLength); }
        else if (lSearchLength >  lReplaceLength) { lResult = Edit_Replace_ByShorter(aSearch, aReplace, lSearchLength, lReplaceLength); }
        else                                      { lResult = Edit_Replace_ByLonger (aSearch, aReplace, lSearchLength, lReplaceLength); }

        return lResult;
    }

    // aSearch [---;R--]
    unsigned int SourceCode::Edit_Search(const char * aSearch)
    {
        if (NULL == aSearch)
        {
            return 0;
        }

        if (NULL == mCode)
        {
            return 0;
        }

        unsigned int lSearchLength = static_cast<unsigned int>(strlen(aSearch));
        if (0 >= lSearchLength)
        {
            return 0;
        }

        const char * lPtr    = mCode;
        unsigned int lResult =     0;

        while (NULL != (lPtr = strstr(lPtr, aSearch)))
        {
            lPtr += lSearchLength;
            lResult++;
        }

        return lResult;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    unsigned int SourceCode::GetArgumentCount() const
    {
        assert( 1 <= mArgumentCount );

        return mArgumentCount;
    }

    const char * SourceCode::GetCode() const
    {
        return mCode;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    // aSearch  [---;R--]
    // aReplace [---;R--]
    unsigned int SourceCode::Edit_Replace_ByEqual(const char * aSearch, const char * aReplace, unsigned int aLength)
    {
        assert(NULL != aSearch );
        assert(NULL != aReplace);
        assert(   0 <  aLength );

        assert(NULL != mCode);

        char       * lPtr    = mCode;
        unsigned int lResult = 0;

        while (NULL != (lPtr = strstr(lPtr, aSearch)))
        {
            memcpy(lPtr, aReplace, aLength);
            lPtr += aLength;
            lResult++;
        }

        return lResult;
    }

    // aSearch  [---;R--]
    // aReplace [---;R--]
    unsigned int SourceCode::Edit_Replace_ByLonger(const char * aSearch, const char * aReplace, unsigned int aSearchLength, unsigned int aReplaceLength)
    {
        assert(NULL          != aSearch       );
        assert(NULL          != aReplace      );
        assert(            1 <= aSearchLength );
        assert(aSearchLength <  aReplaceLength);

        assert(NULL != mCode);

        char       * lPtr    = mCode;
        unsigned int lResult =     0;

        while (NULL != (lPtr = strstr(lPtr, aSearch)))
        {
            *lPtr = '\0';
            lPtr += aSearchLength;
            lResult++;
        }

        mCodeSize_byte += lResult * (aReplaceLength - aSearchLength);

        char * lNewCode = Allocate(mCodeSize_byte);
        assert(NULL != lNewCode);

        char       * lDst = lNewCode;
        const char * lSrc = mCode   ;

        for (unsigned int i = 0; i < lResult; i++)
        {
            strcat_s(lNewCode SIZE_INFO(mCodeSize_byte + 1), lSrc    );
            strcat_s(lNewCode SIZE_INFO(mCodeSize_byte + 1), aReplace);
            lSrc += strlen(lSrc) + aSearchLength;
        }

        strcat_s(lNewCode SIZE_INFO(mCodeSize_byte + 1), lSrc);

        // printf( __CLASS__ "Edit_Replace_ByLonger - delete [] 0x%lx (mCode)\n", reinterpret_cast< uint64_t >( mCode ) );

        delete [] mCode;
        mCode = lNewCode;

        return lResult;
    }

    // aSearch  [---;R--]
    // aReplace [---;R--]
    unsigned int SourceCode::Edit_Replace_ByShorter(const char * aSearch, const char * aReplace, unsigned int aSearchLength, unsigned int aReplaceLength)
    {
        assert(NULL          != aSearch       );
        assert(NULL          != aReplace      );
        assert(            2 <= aSearchLength );
        assert(            1 <= aReplaceLength);
        assert(aSearchLength >  aReplaceLength);

        assert(NULL != mCode);

        char       * lPtr    = mCode;
        unsigned int lResult = 0;

        while (NULL != (lPtr = strstr(lPtr, aSearch)))
        {
            memcpy(lPtr, aReplace, aReplaceLength);
            char * lSrc = lPtr + aSearchLength;
            lPtr += aReplaceLength;
            memmove(lPtr, lSrc, strlen(lSrc) + 1);
            lResult++;
        }

        mCodeSize_byte -= lResult * (aSearchLength - aReplaceLength);

        return lResult;
    }

    void SourceCode::ReleaseCode()
    {
        assert(NULL != mCode         );
        assert(   0 <  mCodeSize_byte);

        // printf( __CLASS__ "ReleaseCode - delete [] 0x%lx (mCode)\n", reinterpret_cast< uint64_t >( mCode ) );

        delete [] mCode;

        mCode          = NULL;
        mCodeSize_byte =    0;
    }

}

// Static functions
/////////////////////////////////////////////////////////////////////////////

char * Allocate(unsigned int aSize_byte)
{
    assert(0 < aSize_byte);

    unsigned int lSize_byte = aSize_byte + 1;

    char * lResult = new char[lSize_byte];
    assert(NULL != lResult);

    memset(lResult, 0, lSize_byte);

    return lResult;
}

#ifdef _KMS_WINDOWS_

    // aOut_byte [---;-W-]

    // NOT TESTED  OpenNet.Filter.ErrorHandling
    //             GetFileSize fail<br>
    //             File too large
    OpenNet::Status GetInputFileSize(HANDLE aHandle, unsigned int * aOut_byte)
    {
        assert(INVALID_HANDLE_VALUE != aHandle  );
        assert(NULL                 != aOut_byte);

        DWORD lSizeHigh;

        (*aOut_byte) = GetFileSize(aHandle, &lSizeHigh);
        if (0 != lSizeHigh)
        {
            return OpenNet::STATUS_INPUT_FILE_TOO_LARGE;
        }

        if (0 >= (*aOut_byte))
        {
            return OpenNet::STATUS_EMPTY_INPUT_FILE;
        }

        return OpenNet::STATUS_OK;
    }

    // aFileName [---;R--]
    // aHandle   [---;-W-]
    OpenNet::Status OpenInputFile(const char * aFileName, HANDLE * aOut)
    {
        assert(NULL != aOut);

        if (NULL == aFileName)
        {
            return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        (*aOut) = CreateFile(aFileName, GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, 0, NULL);
        if (INVALID_HANDLE_VALUE == (*aOut))
        {
            return OpenNet::STATUS_CANNOT_OPEN_INPUT_FILE;
        }

        return OpenNet::STATUS_OK;
    }

    // aOut [---;-W-]

    // NOT TESTED  OpenNet.Filter.ErrorHandling
    //             ReadFile fail<br>
    //             ReadFile do not read the expected size
    OpenNet::Status ReadInputFile(HANDLE aHandle, void * aOut, unsigned int aOutSize_byte)
    {
        assert(INVALID_HANDLE_VALUE != aHandle      );
        assert(NULL                 != aOut         );
        assert(                   0 <  aOutSize_byte);

        DWORD lInfo_byte;

        if (!ReadFile(aHandle, aOut, aOutSize_byte, &lInfo_byte, NULL))
        {
            return OpenNet::STATUS_CANNOT_READ_INPUT_FILE;
        }

        if (aOutSize_byte != lInfo_byte)
        {
            return OpenNet::STATUS_ERROR_READING_INPUT_FILE;
        }

        return OpenNet::STATUS_OK;
    }

#endif
