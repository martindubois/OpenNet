
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Filter.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Windows ============================================================
#include <Windows.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Filter.h>

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static char * Allocate(unsigned int aSize_byte);

static OpenNet::Status GetInputFileSize(HANDLE aHandle, unsigned int * aOut_byte);
static OpenNet::Status OpenInputFile   (const char * aFileName, HANDLE * aOut);
static OpenNet::Status ReadInputFile   (HANDLE aHandle, void * aOut, unsigned int aOutSize_byte);

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    const unsigned int Filter::BUILD_LOG_MAX_SIZE_byte = 64 * 1024;

    Filter::Filter() : mCode(NULL), mCodeLineBuffer(NULL), mCodeLineCount(0), mCodeLines(NULL), mCodeSize_byte(0)
    {
        memset(&mName, 0, sizeof(mName));
    }

    Filter::~Filter()
    {
        Invalidate();

        if (NULL != mCode)
        {
            delete mCode;
        }
    }

    unsigned int Filter::GetCodeLineCount()
    {
        if (0 >= mCodeLineCount)
        {
            CodeLines_Count();
        }

        return mCodeLineCount;
    }

    const char ** Filter::GetCodeLines()
    {
        if (NULL == mCodeLines)
        {
            CodeLines_Generate();
        }

        return mCodeLines;
    }

    unsigned int Filter::GetCodeSize() const
    {
        if (NULL == mCode)
        {
            return 0;
        }

        assert(0 < mCodeSize_byte);

        return mCodeSize_byte;
    }

    const char * Filter::GetName() const
    {
        return mName;
    }

    Status Filter::ResetCode()
    {
        if (NULL == mCode)
        {
            return STATUS_CODE_NOT_SET;
        }

        Invalidate();

        ReleaseCode();

        return STATUS_OK;
    }

    // NOT TESTED  OpenNet.Filter.ErrorHandling
    //             CloseHandle fail<br>
    //             ReadInputFile fail
    Status Filter::SetCode(const char * aFileName)
    {
        if (NULL != mCode)
        {
            return STATUS_CODE_ALREADY_SET;
        }

        assert(0 == mCodeSize_byte);

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
    }

    Status Filter::SetCode(const char * aCode, unsigned int aSize_byte)
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

        mCode = Allocate(aSize_byte);
        assert(NULL != mCode);

        memcpy(mCode, aCode, aSize_byte);

        mCodeSize_byte = aSize_byte;

        return STATUS_OK;
    }

    Status Filter::SetName(const char * aName)
    {
        if (NULL == aName)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        strncpy_s(mName, aName, sizeof(mName) - 1);

        return STATUS_OK;
    }

    void Filter::AddKernelArgs(void * aKernel)
    {
        assert(NULL != aKernel);
    }

    Status Filter::Display(FILE * aOut) const
    {
        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "Filter :\n");

        if (NULL == mCode)
        {
            fprintf(aOut, "  Code not set\n");
        }

        fprintf(aOut, "  Code Size         = %u bytes\n", mCodeSize_byte);
        fprintf(aOut, "  Name              = %s\n"      , mName         );

        if (NULL != mCode)
        {
            fprintf(aOut, "  Code :\n");

            if (NULL == mCodeLines)
            {
                fprintf(aOut, "%s", mCode);
            }
            else
            {
                for (unsigned int i = 0; i < mCodeLineCount; i++)
                {
                    fprintf(aOut, "%3u  %s", i + 1, mCodeLines[i]);
                }
            }

            fprintf(aOut, "\n");
        }

        if (NULL == mBuildLog)
        {
            fprintf(aOut, "  No Build Log\n");
        }
        else
        {
            fprintf(aOut, "  Build Log :\n%s\n", mBuildLog);
        }

        return STATUS_OK;
    }

    unsigned int Filter::Edit_Remove(const char * aSearch)
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

        Invalidate();

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

    unsigned int Filter::Edit_Replace(const char * aSearch, const char * aReplace)
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

        Invalidate();

        unsigned int lResult;

        if      (            0 == lReplaceLength) { lResult = Edit_Remove           (aSearch); }
        else if (lSearchLength == lReplaceLength) { lResult = Edit_Replace_ByEqual  (aSearch, aReplace, lSearchLength); }
        else if (lSearchLength >  lReplaceLength) { lResult = Edit_Replace_ByShorter(aSearch, aReplace, lSearchLength, lReplaceLength); }
        else                                      { lResult = Edit_Replace_ByLonger (aSearch, aReplace, lSearchLength, lReplaceLength); }

        return lResult;
    }

    // aSearch [---;R--]
    unsigned int Filter::Edit_Search(const char * aSearch)
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

    void * Filter::AllocateBuildLog()
    {
        assert(NULL == mBuildLog);

        mBuildLog = new char[BUILD_LOG_MAX_SIZE_byte];
        assert(NULL != mBuildLog);

        return mBuildLog;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    void Filter::CodeLines_Count()
    {
        mCodeLineCount = 0;

        if (NULL != mCode)
        {
            char lLast = 0;

            const char * lPtr = mCode;
            while ('\0' != (*lPtr))
            {
                switch (*lPtr)
                {
                case '\n':
                    if ('\r' != lLast)
                    {
                        mCodeLineCount++;
                        lLast = *lPtr;
                    }
                    break;

                case '\r':
                    if ('\n' != lLast)
                    {
                        mCodeLineCount++;
                        lLast = *lPtr;
                    }
                    break;

                default:
                    lLast = *lPtr;
                }

                lPtr++;
            }

            mCodeLineCount++;
        }
    }

    void Filter::CodeLines_Generate()
    {
        assert(NULL == mCodeLineBuffer);
        assert(NULL == mCodeLines     );

        if (NULL != mCode)
        {
            CodeLines_Count();

            assert(0 < mCodeLineCount);

            mCodeLineBuffer = new       char   [mCodeSize_byte + mCodeLineCount + 1];
            mCodeLines      = new const char * [                 mCodeLineCount    ];

            assert(NULL != mCodeLineBuffer);
            assert(NULL != mCodeLines     );

            char       * lDst  = mCodeLineBuffer;
            unsigned int lLine =               0;
            const char * lSrc  = mCode          ;

            mCodeLines[lLine] = lDst;

            while ('\0' != (*lSrc))
            {
                (*lDst) = (*lSrc); lDst++;

                switch (*lSrc)
                {
                case '\n':
                    if ('\r' == lSrc[1])
                    {
                        (*lDst) = '\r'; lDst++;

                        lSrc ++;
                    }

                    (*lDst) = '\0'; lDst++;

                    lLine++;
                    mCodeLines[lLine] = lDst;
                    break;

                case '\r':
                    if ('\n' == lSrc[1])
                    {
                        (*lDst) = '\n'; lDst++;

                        lSrc ++;
                    }

                    (*lDst) = '\0'; lDst++;

                    lLine++;
                    mCodeLines[lLine] = lDst;
                    break;
                }

                lSrc++;
            }

            (*lDst) = '\0';
        }
    }

    // aSearch  [---;R--]
    // aReplace [---;R--]
    unsigned int Filter::Edit_Replace_ByEqual(const char * aSearch, const char * aReplace, unsigned int aLength)
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
    unsigned int Filter::Edit_Replace_ByLonger(const char * aSearch, const char * aReplace, unsigned int aSearchLength, unsigned int aReplaceLength)
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
            strcat_s(lNewCode, mCodeSize_byte + 1, lSrc    );
            strcat_s(lNewCode, mCodeSize_byte + 1, aReplace);
            lSrc += strlen(lSrc) + aSearchLength;
        }

        strcat_s(lNewCode, mCodeSize_byte + 1, lSrc);

        delete mCode;
        mCode = lNewCode;

        return lResult;
    }

    // aSearch  [---;R--]
    // aReplace [---;R--]
    unsigned int Filter::Edit_Replace_ByShorter(const char * aSearch, const char * aReplace, unsigned int aSearchLength, unsigned int aReplaceLength)
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

    void Filter::Invalidate()
    {
        mCodeLineCount = 0;

        if (NULL != mBuildLog)
        {
            delete mBuildLog;

            mBuildLog = NULL;
        }

        if (NULL != mCodeLines)
        {
            assert(NULL != mCodeLineBuffer);

            delete mCodeLineBuffer;
            delete mCodeLines     ;

            mCodeLineBuffer = NULL;
            mCodeLines      = NULL;
        }
    }

    void Filter::ReleaseCode()
    {
        assert(NULL != mCode         );
        assert(   0 <  mCodeSize_byte);

        delete mCode;

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
