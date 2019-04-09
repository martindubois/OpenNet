
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Internal/UserBuffer_Internal.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "../Component.h"

// ===== OpenNet/Internal ===================================================
#include "../Utils.h"

#include "UserBuffer_Internal.h"

// Public
/////////////////////////////////////////////////////////////////////////////

// ===== OpenNet::UserBuffer ================================================

OpenNet::Status UserBuffer_Internal::Clear()
{
    try
    {
        Clear_Internal();
    }
    catch (KmsLib::Exception * eE)
    {
        return Utl_ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status UserBuffer_Internal::Read(unsigned int aOffset_byte, void * aOut, unsigned int aSize_byte)
{
    assert(0 < mSize_byte);

    if (mSize_byte <= aOffset_byte)
    {
        return OpenNet::STATUS_INVALID_OFFSET;
    }

    if (NULL == aOut)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aSize_byte)
    {
        return OpenNet::STATUS_INVALID_SIZE;
    }

    unsigned int lMaxSize_byte = mSize_byte - aOffset_byte;

    unsigned int lSize_byte = (lMaxSize_byte < aSize_byte) ? lMaxSize_byte : aSize_byte;

    try
    {
        Read_Internal(aOffset_byte, aOut, lSize_byte);
    }
    catch (KmsLib::Exception * eE)
    {
        return Utl_ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

OpenNet::Status UserBuffer_Internal::Write(unsigned int aOffset_byte, const void * aIn, unsigned int aSize_byte)
{
    assert(0 < mSize_byte);

    if (mSize_byte <= aOffset_byte)
    {
        return OpenNet::STATUS_INVALID_OFFSET;
    }

    if (NULL == aIn)
    {
        return OpenNet::STATUS_NOT_ALLOWED_NULL_ARGUMENT;
    }

    if (0 >= aSize_byte)
    {
        return OpenNet::STATUS_INVALID_SIZE;
    }

    unsigned int lMaxSize_byte = mSize_byte - aOffset_byte;

    unsigned int lSize_byte = (lMaxSize_byte < aSize_byte) ? lMaxSize_byte : aSize_byte;

    try
    {
        Write_Internal(aOffset_byte, aIn, lSize_byte);
    }
    catch (KmsLib::Exception * eE)
    {
        return Utl_ExceptionToStatus(eE);
    }

    return OpenNet::STATUS_OK;
}

// Protected
/////////////////////////////////////////////////////////////////////////////

UserBuffer_Internal::UserBuffer_Internal(unsigned int aSize_byte) : mSize_byte(aSize_byte)
{
    assert(0 < aSize_byte);
}
