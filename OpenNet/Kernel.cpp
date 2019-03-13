
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Kernel.cpp

#define __CLASS__     "Kernel::"
#define __NAMESPACE__ "OpenNet::"

// Includes
/////////////////////////////////////////////////////////////////////////////

#include <KmsBase.h>

// ===== C ==================================================================
#include <assert.h>
#include <memory.h>
#include <stdint.h>

// ===== Import/Includes ====================================================
#include <KmsLib/Exception.h>
#include <KmsLib/ValueVector.h>

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Kernel.h>

// ===== Common =============================================================
#include "../Common/OpenNet/Kernel_Statistics.h"

// ===== OpenNet ============================================================
#include "Constants.h"
#include "Event.h"

#ifdef _KMS_LINUX_
    #include "UserBuffer_CUDA.h"
#endif

#ifdef _KMS_WINDOWS_
    #include "OCLW.h"
    #include "UserBuffer_OpenCL.h"
#endif

// Constants
/////////////////////////////////////////////////////////////////////////////

static const OpenNet::StatisticsProvider::StatisticsDescription STATISTICS_DESCRIPTIONS[] =
{
    { "EXECUTION                  ", ""  , 0 }, //  0
    { "EXECUTION - DURATION - AVG ", "us", 1 },
    { "EXECUTION - DURATION - MAX ", "us", 0 },
    { "EXECUTION - DURATION - MIN ", "us", 0 },
    { "QUEUE     - DURATION - AVG ", "us", 1 },
    { "QUEUE     - DURATION - MAX ", "us", 0 }, //  5
    { "QUEUE     - DURATION - MIN ", "us", 0 },
    { "SUBMIT    - DURATION - AVG ", "us", 1 },
    { "SUBMIT    - DURATION - MAX ", "us", 0 },
    { "SUBMIT    - DURATION - MIN ", "us", 0 },

    VALUE_VECTOR_DESCRIPTION_RESERVED, // 10
    VALUE_VECTOR_DESCRIPTION_RESERVED,
    VALUE_VECTOR_DESCRIPTION_RESERVED,

    { "EXECUTION - DURATION - LAST", "us", 0 },
    { "QUEUE     - DURATION - LAST", "us", 0 },
    { "SUBMIT    - DURATION - LAST", "us", 0 }, // 15
};

namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    Kernel::Kernel()
        : StatisticsProvider(STATISTICS_DESCRIPTIONS, KERNEL_STATS_QTY)
        , mBuildLog        (NULL )
        , mCodeLineBuffer  (NULL )
        , mCodeLineCount   (    0)
        , mCodeLines       (NULL )
        , mCommandQueue    (NULL )
        , mProfilingEnabled(false)
        // new ==> delete  See the destructor
        , mStatistics      (new unsigned int [KERNEL_STATS_QTY])
    {
        assert(NULL != mStatistics);

        memset(mStatistics, 0, sizeof(unsigned int) * KERNEL_STATS_QTY);

        Status lStatus = ResetStatistics();
        assert(STATUS_OK == lStatus);
        (void)(lStatus);
    }

    Status Kernel::DisableProfiling()
    {
        if (!mProfilingEnabled)
        {
            return STATUS_PROFILING_ALREADY_DISABLED;
        }

        mProfilingEnabled = false;

        return STATUS_OK;
    }

    Status Kernel::EnableProfiling()
    {
        if (mProfilingEnabled)
        {
            return STATUS_PROFILING_ALREADY_ENABLED;
        }

        mProfilingEnabled = true;

        return STATUS_OK;
    }

    unsigned int Kernel::GetCodeLineCount()
    {
        if (0 >= mCodeLineCount)
        {
            CodeLines_Count();
        }

        return mCodeLineCount;
    }

    const char ** Kernel::GetCodeLines()
    {
        if (NULL == mCodeLines)
        {
            CodeLines_Generate();
        }

        return mCodeLines;
    }

    void * Kernel::GetCommandQueue()
    {
        return mCommandQueue;
    }

    OpenNet::Status Kernel::SetStaticUserArgument(unsigned int aIndex, UserBuffer * aArg)
    {
        if (0 == aIndex)
        {
            return OpenNet::STATUS_INVALID_INDEX;
        }

        if (NULL == aArg)
        {
            mUserArguments.erase(aIndex);
        }
        else
        {
            mUserArguments.insert(UserArgumentMap::value_type(aIndex, aArg));
        }

        return OpenNet::STATUS_OK;
    }

    void Kernel::SetUserKernelArgs(void * aKernel)
    {
        assert(NULL != aKernel);

        for (UserArgumentMap::iterator lIt = mUserArguments.begin(); lIt != mUserArguments.end(); lIt++)
        {
            try
            {
                assert(   0 <  lIt->first );
                assert(NULL != lIt->second);

                #ifdef _KMS_WINDOWS_

                    UserBuffer_OpenCL * lUB = dynamic_cast<UserBuffer_OpenCL *>(lIt->second);
                    assert(NULL != lUB       );
                    assert(NULL != lUB->mMem );

                    OCLW_SetKernelArg(reinterpret_cast<cl_kernel>(aKernel), lIt->first, sizeof(cl_mem), &lUB->mMem);

                #endif
            }
            catch (KmsLib::Exception * eE)
            {
                eE->Write(stderr);
            }
        }
    }

    void Kernel::SetUserKernelArgs(void * * aArguments)
    {
        assert(NULL != aArguments);

        for (UserArgumentMap::iterator lIt = mUserArguments.begin(); lIt != mUserArguments.end(); lIt++)
        {
            try
            {
                assert(   0 <  lIt->first );
                assert(NULL != lIt->second);

                #ifdef _KMS_LINUX_

                    UserBuffer_CUDA * lUB = dynamic_cast<UserBuffer_CUDA *>(lIt->second);
                    assert(NULL != lUB          );
                    assert(   0 != lUB->mMem_DA );

                    aArguments[ lIt->first ] = & lUB->mMem_DA;

                #endif
            }
            catch (KmsLib::Exception * eE)
            {
                eE->Write(stderr);
            }
        }
    }

    // ===== SourceCode =====================================================

    Kernel::~Kernel()
    {
        assert( NULL != mStatistics );

        Invalidate();

        // new ==> delete  See the constructor
        delete [] mStatistics;
    }

    Status Kernel::AppendCode(const char * aCode, unsigned int aCodeSize_byte)
    {
        Status lResult = SourceCode::AppendCode(aCode, aCodeSize_byte);
        if (STATUS_OK == lResult)
        {
            Invalidate();
        }

        return lResult;
    }

    Status Kernel::ResetCode()
    {
        Status lResult = SourceCode::ResetCode();
        if (STATUS_OK == lResult)
        {
            Invalidate();
        }

        return lResult;
    }

    Status Kernel::SetCode(const char * aFileName)
    {
        Status lResult = SourceCode::SetCode(aFileName);
        if (STATUS_OK == lResult)
        {
            Invalidate();
        }

        return lResult;
    }

    Status Kernel::SetCode(const char * aCode, unsigned int aCodeSize_byte)
    {
        Status lResult = SourceCode::SetCode(aCode, aCodeSize_byte);
        if (STATUS_OK == lResult)
        {
            Invalidate();
        }

        return lResult;
    }

    Status Kernel::Display(FILE * aOut) const
    {
        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "  Kernel :\n");
        fprintf(aOut, "    Profiling Enabled = %s\n", mProfilingEnabled ? "true" : "false");

        if (NULL == mBuildLog)
        {
            fprintf(aOut, "    No Build Log\n");
        }
        else
        {
            fprintf(aOut, "    Build Log         =\n%s\n", mBuildLog);
        }

        if (NULL != mCodeLines)
        {
            fprintf(aOut, "    Code Lines        =\n");

            for (unsigned int i = 0; i < mCodeLineCount; i++)
            {
                fprintf(aOut, "%3u  %s", i + 1, mCodeLines[i]);
            }

            fprintf(aOut, "\n");
        }

        return SourceCode::Display(aOut);
    }

    unsigned int Kernel::Edit_Remove(const char * aSearch)
    {
        unsigned int lResult = SourceCode::Edit_Remove(aSearch);
        if (0 < lResult)
        {
            Invalidate();
        }

        return lResult;
    }

    unsigned int Kernel::Edit_Replace(const char * aSearch, const char * aReplace)
    {
        unsigned int lResult = SourceCode::Edit_Replace(aSearch, aReplace);
        if (0 < lResult)
        {
            Invalidate();
        }

        return lResult;
    }

    // ===== StatisticsProvider =============================================

    Status Kernel::GetStatistics(unsigned int * aOut, unsigned int aOutSize_byte, unsigned int * aInfo_byte, bool aReset)
    {
        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        if (0 >= aOutSize_byte)
        {
            return STATUS_BUFFER_TOO_SMALL;
        }

        unsigned int lCount = aOutSize_byte / sizeof(unsigned int);

        if (lCount > KERNEL_STATS_QTY)
        {
            lCount = KERNEL_STATS_QTY;
        }

        unsigned int lSize_byte = lCount * sizeof(unsigned int);

        memcpy(aOut, mStatistics, lSize_byte);

        if (aReset)
        {
            ResetStatistics();
        }

        if (NULL != aInfo_byte)
        {
            (*aInfo_byte) = lSize_byte;
        }

        return STATUS_OK;
    }

    Status Kernel::ResetStatistics()
    {
        memset(mStatistics, 0, KERNEL_STATS_RESET_QTY * sizeof(unsigned int));

        mStatistics[KERNEL_STATS_EXECUTION_DURATION_MIN_us] = 0xffffffff;
        mStatistics[KERNEL_STATS_QUEUE_DURATION_MIN_us    ] = 0xffffffff;
        mStatistics[KERNEL_STATS_SUBMIT_DURATION_MIN_us   ] = 0xffffffff;

        memset(&mStatisticsSums, 0, sizeof(mStatisticsSums));

        return STATUS_OK;
    }

    // Internal
    /////////////////////////////////////////////////////////////////////////

    void Kernel::ResetCommandQueue()
    {
        assert(NULL != mCommandQueue);

        mCommandQueue = NULL;
    }

    void Kernel::SetCommandQueue(void * aCommandQueue)
    {
        assert(NULL != aCommandQueue);

        assert(NULL == mCommandQueue);

        mCommandQueue = aCommandQueue;
    }

    // aEvent [---;R--]

    // CRITICAL PATH  Processing.Profiling
    //                1 / iteration
    void Kernel::AddStatistics( Event * aEvent )
    {
        uint64_t lLast[3];

        lLast[0] = aEvent->GetQueued   ();
        lLast[1] = aEvent->GetSubmitted();
        lLast[2] = aEvent->GetExecution();

        for (unsigned int i = 0; i < 3; i++)
        {
            mStatisticsSums[i] += lLast[i];
        }

        mStatistics[KERNEL_STATS_EXECUTION] ++;

        mStatistics[KERNEL_STATS_EXECUTION_DURATION_AVG_us ] = static_cast<unsigned int>(mStatisticsSums[2] / mStatistics[KERNEL_STATS_EXECUTION]);
        mStatistics[KERNEL_STATS_EXECUTION_DURATION_LAST_us] = static_cast<unsigned int>(lLast[2]);

        if (mStatistics[KERNEL_STATS_EXECUTION_DURATION_MAX_us] < lLast[2]) { mStatistics[KERNEL_STATS_EXECUTION_DURATION_MAX_us] = static_cast<unsigned int>(lLast[2]); }
        if (mStatistics[KERNEL_STATS_EXECUTION_DURATION_MIN_us] > lLast[2]) { mStatistics[KERNEL_STATS_EXECUTION_DURATION_MIN_us] = static_cast<unsigned int>(lLast[2]); }

        mStatistics[KERNEL_STATS_QUEUE_DURATION_AVG_us ] = static_cast<unsigned int>(mStatisticsSums[0] / mStatistics[KERNEL_STATS_EXECUTION]);
        mStatistics[KERNEL_STATS_QUEUE_DURATION_LAST_us] = static_cast<unsigned int>(lLast[0]);

        if (mStatistics[KERNEL_STATS_QUEUE_DURATION_MAX_us] < lLast[0]) { mStatistics[KERNEL_STATS_QUEUE_DURATION_MAX_us] = static_cast<unsigned int>(lLast[0]); }
        if (mStatistics[KERNEL_STATS_QUEUE_DURATION_MIN_us] > lLast[0]) { mStatistics[KERNEL_STATS_QUEUE_DURATION_MIN_us] = static_cast<unsigned int>(lLast[0]); }

        mStatistics[KERNEL_STATS_SUBMIT_DURATION_AVG_us ] = static_cast<unsigned int>(mStatisticsSums[1] / mStatistics[KERNEL_STATS_EXECUTION]);
        mStatistics[KERNEL_STATS_SUBMIT_DURATION_LAST_us] = static_cast<unsigned int>(lLast[1]);

        if (mStatistics[KERNEL_STATS_SUBMIT_DURATION_MAX_us] < lLast[1]) { mStatistics[KERNEL_STATS_SUBMIT_DURATION_MAX_us] = static_cast<unsigned int>(lLast[1]); }
        if (mStatistics[KERNEL_STATS_SUBMIT_DURATION_MIN_us] > lLast[1]) { mStatistics[KERNEL_STATS_SUBMIT_DURATION_MIN_us] = static_cast<unsigned int>(lLast[1]); }
    }

    // Kernel::AllocateBuildLog ==> Kernel::Invalidate
    void * Kernel::AllocateBuildLog()
    {
        assert(NULL == mBuildLog);

        // new ==> delete  See Kernel::Invalidate
        mBuildLog = new char[BUILD_LOG_MAX_SIZE_byte];
        assert(NULL != mBuildLog);

        return mBuildLog;
    }

    // Kernel::AllocateBuildLog ==> Kernel::Invalidate
    char * Kernel::AllocateBuildLog( size_t aSize_byte )
    {
        assert( 0 < aSize_byte );

        assert(NULL == mBuildLog);

        // new ==> delete  See Kernel::Invalidate
        mBuildLog = new char[ aSize_byte ];
        assert( NULL != mBuildLog );

        return mBuildLog;
    }

    // Private
    /////////////////////////////////////////////////////////////////////////

    void Kernel::CodeLines_Count()
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

    // Kernel::CodeLine_Generate ==> Kernel::Invalidate
    void Kernel::CodeLines_Generate()
    {
        assert(NULL == mCodeLineBuffer);
        assert(NULL == mCodeLines     );

        if (NULL != mCode)
        {
            CodeLines_Count();

            assert(0 < mCodeLineCount);

            // new ==> delete  See Invalidate
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

    // Kernel::AllocateBuildLog   ==> Kernel::Invalidate
    // Kernel::CodeLines_Generate ==> Kernel::Invalidate
    void Kernel::Invalidate()
    {
        mCodeLineCount = 0;

        if (NULL != mBuildLog)
        {
            // new ==> delete  See AllocateBuildLog
            delete[] mBuildLog;

            mBuildLog = NULL;
        }

        if (NULL != mCodeLines)
        {
            assert(NULL != mCodeLineBuffer);

            // new ==> delete  See CodeLines_Generate
            delete[] mCodeLineBuffer;
            delete[] mCodeLines;

            mCodeLineBuffer = NULL;
            mCodeLines = NULL;
        }
    }

}
