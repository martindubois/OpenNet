
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/StatisticsProvider.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <assert.h>

// ===== Import/Includes ====================================================
#include <KmsLib/ValueVector.h>

// ===== Includes ===========================================================
#include <OpenNet/StatisticsProvider.h>


namespace OpenNet
{

    // Public
    /////////////////////////////////////////////////////////////////////////

    const StatisticsProvider::StatisticsDescription * StatisticsProvider::GetStatisticsDescriptions() const
    {
        return mStatisticsDescriptions;
    }

    Status StatisticsProvider::DisplayStatistics(const unsigned int * aIn, unsigned int aInSize_byte, FILE * aOut, unsigned int aMinLevel)
    {
        if (NULL == aIn)
        {
            return STATUS_INVALID_REFERENCE;
        }

        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        unsigned int lCount = aInSize_byte / sizeof(unsigned int);

        if (mStatisticsQty < lCount)
        {
            lCount = mStatisticsQty;
        }

        fprintf(aOut, "Statistics:\n");
        KmsLib::ValueVector::Display(aIn, lCount, VALUE_VECTOR_DISPLAY_FLAG_HIDE_ZERO, aOut, reinterpret_cast<const KmsLib::ValueVector::Description *>(mStatisticsDescriptions), aMinLevel);

        return STATUS_OK;
    }

    // Protected
    /////////////////////////////////////////////////////////////////////////

    StatisticsProvider::StatisticsProvider(const StatisticsDescription * aStatisticsDescriptions, unsigned int aStatisticsQty)
        : mStatisticsDescriptions(aStatisticsDescriptions)
        , mStatisticsQty         (aStatisticsQty         )
    {
    }

}
