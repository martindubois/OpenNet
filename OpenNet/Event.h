
// Author     KMS - Martin Dubois, ing.
// Copywrite  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet/Event.h

#pragma once

// Class
/////////////////////////////////////////////////////////////////////////////

class Event
{

public:

    virtual void Init( bool aProfiling );

    uint64_t GetExecution() const;
    uint64_t GetQueued   () const;
    uint64_t GetSubmitted() const;

    // Wait until the end of the processing
    //
    // Exception  KmsLib::Exception *
    // Thread     Worker

    // CRITICAL PATH  Processing
    //                1 / iteration
    virtual void Wait() = 0;

protected:

    Event( bool aProfilint = false );

    bool mProfilling;

    uint64_t mExecution_us;
    uint64_t mQueued_us   ;
    uint64_t mSubmitted_us;

};

// Public
/////////////////////////////////////////////////////////////////////////////

// CRITICAL PATH  Processing.Profiling
//                1 / iteration
inline uint64_t Event::GetExecution() const
{
    return mExecution_us;
}

// CRITICAL PATH  Processing.Profiling
//                1 / iteration
inline uint64_t Event::GetQueued() const
{
    return mQueued_us;
}

// CRITICAL PATH  Processing.Profiling
//                1 / iteration
inline uint64_t Event::GetSubmitted() const
{
    return mSubmitted_us;
}
