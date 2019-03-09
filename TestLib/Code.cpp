
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2019 KMS. All rights reserved
// Product    OpenNet
// File       TestLib/Code.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== C ==================================================================
#include <stdlib.h>

// ===== TestLib ============================================================
#include "Code.h"

// Static constants
/////////////////////////////////////////////////////////////////////////////

#define EOL "\n"

static const char * FUNCTION_DO_NOT_REPLY_ON_ERROR_0 =
"OPEN_NET_FUNCTION_DECLARE( DoNotReplyOnError0 )"                                          EOL
"{"                                                                                        EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                              EOL
                                                                                           EOL
"        OPEN_NET_GLOBAL unsigned short * lData;"                                          EOL
                                                                                           EOL
"        unsigned int lResult = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );"       EOL
"        unsigned int i;"                                                                  EOL
                                                                                           EOL
"        lData = (OPEN_NET_GLOBAL unsigned short *)( lBase + lPacketInfo->mOffset_byte );" EOL
                                                                                           EOL
"        for ( i = 0; i < 3; i ++)"                                                        EOL
"        {"                                                                                EOL
"            if ( 0xffff != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult = OPEN_NET_PACKET_PROCESSED;"                                     EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        if ( 0x0a0a != lData[ 6 ] )"                                                      EOL
"        {"                                                                                EOL
"            lResult = OPEN_NET_PACKET_PROCESSED;"                                         EOL
"        }"                                                                                EOL
                                                                                           EOL
"        for ( i = 7; i < ( lPacketInfo->mSize_byte / sizeof( unsigned short ) ); i ++)"   EOL
"        {"                                                                                EOL
"            if ( 0x0000 != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult = OPEN_NET_PACKET_PROCESSED;"                                     EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        lPacketInfo->mSendTo = lResult;"                                                  EOL
                                                                                           EOL
"    OPEN_NET_FUNCTION_END"                                                                EOL
"}"                                                                                        EOL;

static const char * FUNCTION_DO_NOT_REPLY_ON_ERROR_1 =
"OPEN_NET_FUNCTION_DECLARE( DoNotReplyOnError1 )"                                          EOL
"{"                                                                                        EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                              EOL
                                                                                           EOL
"        OPEN_NET_GLOBAL unsigned short * lData;"                                          EOL
                                                                                           EOL
"        unsigned int lResult = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );"       EOL
"        unsigned int i;"                                                                  EOL
                                                                                           EOL
"        lData = (OPEN_NET_GLOBAL unsigned short *)( lBase + lPacketInfo->mOffset_byte );" EOL
                                                                                           EOL
"        for ( i = 0; i < 3; i ++)"                                                        EOL
"        {"                                                                                EOL
"            if ( 0xffff != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult = OPEN_NET_PACKET_PROCESSED;"                                     EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        if ( 0x0a0a != lData[ 6 ] )"                                                      EOL
"        {"                                                                                EOL
"            lResult = OPEN_NET_PACKET_PROCESSED;"                                         EOL
"        }"                                                                                EOL
                                                                                           EOL
"        for ( i = 7; i < ( lPacketInfo->mSize_byte / sizeof( unsigned short ) ); i ++)"   EOL
"        {"                                                                                EOL
"            if ( 0x0000 != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult = OPEN_NET_PACKET_PROCESSED;"                                     EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        lPacketInfo->mSendTo = lResult;"                                                  EOL
                                                                                           EOL
"    OPEN_NET_FUNCTION_END"                                                                EOL
"}"                                                                                        EOL;

static const char * FUNCTION_FORWARD_0 =
"OPEN_NET_FUNCTION_DECLARE( Forward0 )"                                              EOL
"{"                                                                                  EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                        EOL
                                                                                     EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );" EOL
                                                                                     EOL
"    OPEN_NET_FUNCTION_END"                                                          EOL
"}"                                                                                  EOL;

static const char * FUNCTION_FORWARD_1 =
"OPEN_NET_FUNCTION_DECLARE( Forward1 )"                                              EOL
"{"                                                                                  EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                        EOL
                                                                                     EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );" EOL
                                                                                     EOL
"    OPEN_NET_FUNCTION_END"                                                          EOL
"}"                                                                                  EOL;

static const char * FUNCTION_NOTHING_0 =
"OPEN_NET_FUNCTION_DECLARE( Nothing0 )"                     EOL
"{"                                                         EOL
"    OPEN_NET_FUNCTION_BEGIN"                               EOL
                                                            EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED;" EOL
                                                            EOL
"    OPEN_NET_FUNCTION_END"                                 EOL
"}"                                                         EOL;

static const char * FUNCTION_NOTHING_1 =
"OPEN_NET_FUNCTION_DECLARE( Nothing1 )"                     EOL
"{"                                                         EOL
"    OPEN_NET_FUNCTION_BEGIN"                               EOL
                                                            EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED;" EOL
                                                            EOL
"    OPEN_NET_FUNCTION_END"                                 EOL
"}"                                                         EOL;

static const char * FUNCTION_REPLY_ON_ERROR_0 =
"OPEN_NET_FUNCTION_DECLARE( ReplyOnError0 )"                                               EOL
"{"                                                                                        EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                              EOL
                                                                                           EOL
"        OPEN_NET_GLOBAL unsigned short * lData;"                                          EOL
"        unsigned int                     lResult = OPEN_NET_PACKET_PROCESSED;"            EOL
"        unsigned int                     i;"                                              EOL
                                                                                           EOL
"        lData = (OPEN_NET_GLOBAL unsigned short *)( lBase + lPacketInfo->mOffset_byte );" EOL
                                                                                           EOL
"        for ( i = 0; i < 3; i ++)"                                                        EOL
"        {"                                                                                EOL
"            if ( 0xffff != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                           EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        if ( 0x0a0a != lData[ 6 ] )"                                                      EOL
"        {"                                                                                EOL
"            lResult |= 1 << ADAPTER_INDEX;"                                               EOL
"        }"                                                                                EOL
                                                                                           EOL
"        for ( i = 7; i < ( lPacketInfo->mSize_byte / sizeof( unsigned short ) ); i ++ )"  EOL
"        {"                                                                                EOL
"            if ( 0x0000 != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                           EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        lPacketInfo->mSendTo = lResult;"                                                  EOL
                                                                                           EOL
"    OPEN_NET_FUNCTION_END"                                                                EOL
"}"                                                                                        EOL;

static const char * FUNCTION_REPLY_ON_ERROR_1 =
"OPEN_NET_FUNCTION_DECLARE( ReplyOnError1 )"                                               EOL
"{"                                                                                        EOL
"    OPEN_NET_FUNCTION_BEGIN"                                                              EOL
                                                                                           EOL
"        OPEN_NET_GLOBAL unsigned short * lData;"                                          EOL
"        unsigned int                     lResult = OPEN_NET_PACKET_PROCESSED;"            EOL
"        unsigned int                     i;"                                              EOL
                                                                                           EOL
"        lData = (OPEN_NET_GLOBAL unsigned short *)( lBase + lPacketInfo->mOffset_byte );" EOL
                                                                                           EOL
"        for ( i = 0; i < 3; i ++)"                                                        EOL
"        {"                                                                                EOL
"            if ( 0xffff != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                           EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        if ( 0x0a0a != lData[ 6 ] )"                                                      EOL
"        {"                                                                                EOL
"            lResult |= 1 << ADAPTER_INDEX;"                                               EOL
"        }"                                                                                EOL
                                                                                           EOL
"        for ( i = 7; i < ( lPacketInfo->mSize_byte / sizeof( unsigned short ) ); i ++ )"  EOL
"        {"                                                                                EOL
"            if ( 0x0000 != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                           EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        lPacketInfo->mSendTo = lResult;"                                                  EOL
                                                                                           EOL
"    OPEN_NET_FUNCTION_END"                                                                EOL
"}"                                                                                        EOL;

static const char * KERNEL_DO_NOT_REPLY_ON_ERROR =
"#include <OpenNetK/Kernel.h>"                                                             EOL
                                                                                           EOL
"OPEN_NET_KERNEL_DECLARE"                                                                  EOL
"{"                                                                                        EOL
"    OPEN_NET_KERNEL_BEGIN"                                                                EOL
                                                                                           EOL
"        OPEN_NET_GLOBAL unsigned short * lData;"                                          EOL
                                                                                           EOL
"        unsigned int lResult = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );"       EOL
"        unsigned int i;"                                                                  EOL
                                                                                           EOL
"        lData = (OPEN_NET_GLOBAL unsigned short *)( lBase + lPacketInfo->mOffset_byte );" EOL
                                                                                           EOL
"        for ( i = 0; i < 3; i ++)"                                                        EOL
"        {"                                                                                EOL
"            if ( 0xffff != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult = OPEN_NET_PACKET_PROCESSED;"                                     EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        if ( 0x0a0a != lData[ 6 ] )"                                                      EOL
"        {"                                                                                EOL
"            lResult = OPEN_NET_PACKET_PROCESSED;"                                         EOL
"        }"                                                                                EOL
                                                                                           EOL
"        for ( i = 7; i < ( lPacketInfo->mSize_byte / sizeof( unsigned short ) ); i ++ )"  EOL
"        {"                                                                                EOL
"            if ( 0x0000 != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult = OPEN_NET_PACKET_PROCESSED;"                                     EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        lPacketInfo->mSendTo = lResult;"                                                  EOL
                                                                                           EOL
"    OPEN_NET_KERNEL_END"                                                                  EOL
"}"                                                                                        EOL;

static const char * KERNEL_FORWARD =
"#include <OpenNetK/Kernel.h>"                                                       EOL
                                                                                     EOL
"OPEN_NET_KERNEL_DECLARE"                                                            EOL
"{"                                                                                  EOL
"    OPEN_NET_KERNEL_BEGIN"                                                          EOL
                                                                                     EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED | ( 1 << ADAPTER_INDEX );" EOL
                                                                                     EOL
"    OPEN_NET_KERNEL_END"                                                            EOL
"}"                                                                                  EOL;

static const char * KERNEL_NOTHING =
"#include <OpenNetK/Kernel.h>"                              EOL
                                                            EOL
"OPEN_NET_KERNEL_DECLARE"                                   EOL
"{"                                                         EOL
"    OPEN_NET_KERNEL_BEGIN"                                 EOL
                                                            EOL
"        lPacketInfo->mSendTo = OPEN_NET_PACKET_PROCESSED;" EOL
                                                            EOL
"    OPEN_NET_KERNEL_END"                                   EOL
"}"                                                         EOL;

static const char * KERNEL_REPLY_ON_ERROR =
"#include <OpenNetK/Kernel.h>"                                                             EOL
                                                                                           EOL
"OPEN_NET_KERNEL_DECLARE"                                                                  EOL
"{"                                                                                        EOL
"    OPEN_NET_KERNEL_BEGIN"                                                                EOL
                                                                                           EOL
"        OPEN_NET_GLOBAL unsigned short * lData;"                                          EOL
"        unsigned int                     lResult = OPEN_NET_PACKET_PROCESSED;"            EOL
"        unsigned int                     i;"                                              EOL
                                                                                           EOL
"        lData = (OPEN_NET_GLOBAL unsigned short *)( lBase + lPacketInfo->mOffset_byte );" EOL
                                                                                           EOL
"        for ( i = 0; i < 3; i ++)"                                                        EOL
"        {"                                                                                EOL
"            if ( 0xffff != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                           EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        if ( 0x0a0a != lData[ 6 ] )"                                                      EOL
"        {"                                                                                EOL
"            lResult |= 1 << ADAPTER_INDEX;"                                               EOL
"        }"                                                                                EOL
                                                                                           EOL
"        for ( i = 7; i < ( lPacketInfo->mSize_byte / sizeof( unsigned short ) ); i ++ )"  EOL
"        {"                                                                                EOL
"            if ( 0x0000 != lData[ i ] )"                                                  EOL
"            {"                                                                            EOL
"                lResult |= 1 << ADAPTER_INDEX;"                                           EOL
"            }"                                                                            EOL
"        }"                                                                                EOL
                                                                                           EOL
"        lPacketInfo->mSendTo = lResult;"                                                  EOL
                                                                                           EOL
"    OPEN_NET_KERNEL_END"                                                                  EOL
"}"                                                                                        EOL;

static const char * KERNEL_REPLY_ON_SEQUENCE_ERROR =
"#include <OpenNetK/Kernel.h>"                                                                                                 EOL
                                                                                                                               EOL
"OPEN_NET_KERNEL void Filter( OPEN_NET_GLOBAL OpenNet_BufferHeader * aBufferHeader, OPEN_NET_GLOBAL unsigned int * aCounter )" EOL
"{"                                                                                                                            EOL
"    OPEN_NET_KERNEL_BEGIN"                                                                                                    EOL
                                                                                                                               EOL
"        OPEN_NET_GLOBAL unsigned int * lData;"                                                                                EOL
"        unsigned int                   lResult = OPEN_NET_PACKET_PROCESSED;"                                                  EOL
                                                                                                                               EOL
"        lData = (OPEN_NET_GLOBAL unsigned int *)( lBase + lPacketInfo->mOffset_byte );"                                       EOL
                                                                                                                               EOL
"        if ( aCounter[ 0 ] > lData[ 4 ] )"                                                                                    EOL
"        {"                                                                                                                    EOL
"             lResult |= 1 << OPEN_NET_ADAPTER_NO;"                                                                            EOL
"        }"                                                                                                                    EOL
                                                                                                                               EOL
"        OPEN_NET_GLOBAL_MEMORY_FENCE;"                                                                                        EOL
                                                                                                                               EOL
"        aCounter[ OPEN_NET_PACKET_INDEX ] = lData[ 4 ];"                                                                      EOL
                                                                                                                               EOL
"        OPEN_NET_GLOBAL_MEMORY_FENCE;"                                                                                        EOL
                                                                                                                               EOL
"        if ( 0 == ( OPEN_NET_PACKET_INDEX % 2 ) )"                                                                            EOL
"        {"                                                                                                                    EOL
"            if ( aCounter[ OPEN_NET_PACKET_INDEX ] < aCounter[ OPEN_NET_PACKET_INDEX + 1 ] )"                                 EOL
"            {"                                                                                                                EOL
"                aCounter[ OPEN_NET_PACKET_INDEX ] = aCounter[ OPEN_NET_PACKET_INDEX + 1 ];"                                   EOL
"            }"                                                                                                                EOL
"        }"                                                                                                                    EOL
                                                                                                                               EOL
"        if ( 0 == ( OPEN_NET_PACKET_INDEX % 4 ) )"                                                                            EOL
"        {"                                                                                                                    EOL
"            if ( aCounter[ OPEN_NET_PACKET_INDEX ] < aCounter[ OPEN_NET_PACKET_INDEX + 2 ] )"                                 EOL
"            {"                                                                                                                EOL
"                aCounter[ OPEN_NET_PACKET_INDEX ] = aCounter[ OPEN_NET_PACKET_INDEX + 2 ];"                                   EOL
"            }"                                                                                                                EOL
"        }"                                                                                                                    EOL
                                                                                                                               EOL
"        if ( 0 == ( OPEN_NET_PACKET_INDEX % 8 ) )"                                                                            EOL
"        {"                                                                                                                    EOL
"            if ( aCounter[ OPEN_NET_PACKET_INDEX ] < aCounter[ OPEN_NET_PACKET_INDEX + 4 ] )"                                 EOL
"            {"                                                                                                                EOL
"                aCounter[ OPEN_NET_PACKET_INDEX ] = aCounter[ OPEN_NET_PACKET_INDEX + 4 ];"                                   EOL
"            }"                                                                                                                EOL
"        }"                                                                                                                    EOL
                                                                                                                               EOL
"        if ( 0 == ( OPEN_NET_PACKET_INDEX % 16 ) )"                                                                           EOL
"        {"                                                                                                                    EOL
"            if ( aCounter[ OPEN_NET_PACKET_INDEX ] < aCounter[ OPEN_NET_PACKET_INDEX + 8 ] )"                                 EOL
"            {"                                                                                                                EOL
"                aCounter[ OPEN_NET_PACKET_INDEX ] = aCounter[ OPEN_NET_PACKET_INDEX + 8 ];"                                   EOL
"            }"                                                                                                                EOL
"        }"                                                                                                                    EOL
                                                                                                                               EOL
"        if ( 0 == ( OPEN_NET_PACKET_INDEX % 32 ) )"                                                                           EOL
"        {"                                                                                                                    EOL
"            if ( aCounter[ OPEN_NET_PACKET_INDEX ] < aCounter[ OPEN_NET_PACKET_INDEX + 16 ] )"                                EOL
"            {"                                                                                                                EOL
"                aCounter[ OPEN_NET_PACKET_INDEX ] = aCounter[ OPEN_NET_PACKET_INDEX + 16 ];"                                  EOL
"            }"                                                                                                                EOL
"        }"                                                                                                                    EOL
                                                                                                                               EOL
"        if ( ( 64 <= OPEN_NET_PACKET_QTY ) && ( 0 == ( OPEN_NET_PACKET_INDEX % 64 ) ) )"                                      EOL
"        {"                                                                                                                    EOL
"            if ( aCounter[ OPEN_NET_PACKET_INDEX ] < aCounter[ OPEN_NET_PACKET_INDEX + 32 ] )"                                EOL
"            {"                                                                                                                EOL
"                aCounter[ OPEN_NET_PACKET_INDEX ] = aCounter[ OPEN_NET_PACKET_INDEX + 32 ];"                                  EOL
"            }"                                                                                                                EOL
"        }"                                                                                                                    EOL
                                                                                                                               EOL
"        lPacketInfo->mSendTo = lResult;"                                                                                      EOL
                                                                                                                               EOL
"    OPEN_NET_KERNEL_END"                                                                                                      EOL
"}";

// Global Constants
/////////////////////////////////////////////////////////////////////////////

const CodeInfo CODES[TestLib::CODE_QTY] =
{
    { "DEFAULT"                , NULL                          , { NULL                            , NULL                             }, { NULL                , NULL                 } },
    { "DO_NOT_REPLY_ON_ERROR"  , KERNEL_DO_NOT_REPLY_ON_ERROR  , { FUNCTION_DO_NOT_REPLY_ON_ERROR_0, FUNCTION_DO_NOT_REPLY_ON_ERROR_1 }, { "DoNotReplyOnError0", "DoNotReplyOnError1" } },
    { "FORWARD"                , KERNEL_FORWARD                , { FUNCTION_FORWARD_0              , FUNCTION_FORWARD_1               }, { "Forward0"          , "Forward1"           } },
    { "NONE"                   , NULL                          , { NULL                            , NULL                             }, { NULL                , NULL                 } },
    { "NOTHING"                , KERNEL_NOTHING                , { FUNCTION_NOTHING_0              , FUNCTION_NOTHING_1               }, { "Nothing0"          , "Nothing1"           } },
    { "REPLY"                  , KERNEL_FORWARD                , { FUNCTION_FORWARD_0              , FUNCTION_FORWARD_1               }, { "Forward0"          , "Forward1"           } },
    { "REPLY_ON_ERROR"         , KERNEL_REPLY_ON_ERROR         , { FUNCTION_REPLY_ON_ERROR_0       , FUNCTION_REPLY_ON_ERROR_1        }, { "ReplyOnError0"     , "ReplyOnError1"      } },
    { "REPLY_ON_SEQUENCE_ERROR", KERNEL_REPLY_ON_SEQUENCE_ERROR, { NULL                            , NULL                             }, { NULL                , NULL                 } },
};
