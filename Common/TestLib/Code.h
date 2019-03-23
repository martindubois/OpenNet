
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved.
// Product    OpenNet
// File       Common/TestLib/Code.h

#pragma once

namespace TestLib
{

    // Data types
    /////////////////////////////////////////////////////////////////////////

    typedef enum
    {
        CODE_DEFAULT                ,
        CODE_DO_NOT_REPLY_ON_ERROR  ,
        CODE_FORWARD                ,
        CODE_NONE                   ,
        CODE_NOTHING                ,
        CODE_REPLY                  ,
        CODE_REPLY_ON_ERROR         ,
        CODE_REPLY_ON_SEQUENCE_ERROR,
        CODE_SIGNAL_EVENT           ,

        CODE_QTY
    }
    Code;

}
