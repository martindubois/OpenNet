
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Test/OpenNetK.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// Macros
/////////////////////////////////////////////////////////////////////////////

#define OPEN_NET_CONSTANT const
#define OPEN_NET_DEVICE
#define OPEN_NET_GLOBAL

// Includes - Part 2
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================

#include <OpenNetK/Types.h>

#include <OpenNetK/ARP.h>
#include <OpenNetK/ByteOrder.h>
#include <OpenNetK/Ethernet.h>
#include <OpenNetK/IPv4.h>
#include <OpenNetK/IPv6.h>
#include <OpenNetK/RegEx.h>
#include <OpenNetK/TCP.h>
#include <OpenNetK/UDP.h>

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static int RegEx_TestString    (RegEx * aRegEx, const char * aStr);
static int RegEx_TestString_0  (RegEx * aRegEx, const char * aStr);
static int RegEx_TestString_End(RegEx * aRegEx, const char * aStr);

// Tests
/////////////////////////////////////////////////////////////////////////////

KMS_TEST_BEGIN(OpenNetK_Base)
{
    unsigned char lBuffer[] = { 0x45, 2, 3, 4, 5, 6, 7, 8, 9, 0xa, 0xb, 0xc, 0xd, 0xe };

    // ===== Arp.h ==========================================================

    KMS_TEST_COMPARE(0x0403, ARP_Protocol(lBuffer));

    KMS_TEST_ASSERT(reinterpret_cast<unsigned short *>(lBuffer + 24) == ARP_Destination(lBuffer));
    KMS_TEST_ASSERT(reinterpret_cast<unsigned short *>(lBuffer + 14) == ARP_Source     (lBuffer));

    // ===== ByteOrder.h ====================================================
    KMS_TEST_COMPARE(0x0201    , ByteOrder_Swap16(0x0102    ));
    KMS_TEST_COMPARE(0x04030201, ByteOrder_Swap32(0x01020304));

    // ===== Ethernet.h =====================================================

    OpenNet_PacketInfo lPacketInfo;

    lPacketInfo.mOffset_byte =  0;
    lPacketInfo.mSize_byte   = 20;

    KMS_TEST_COMPARE(0x0e0d, Ethernet_Type   (lBuffer, &lPacketInfo));
    KMS_TEST_COMPARE(     0, Ethernet_Vlan   (lBuffer, &lPacketInfo));
    KMS_TEST_COMPARE(     0, Ethernet_VlanTag(lBuffer, &lPacketInfo));

    KMS_TEST_ASSERT(                                  (lBuffer + 14) == Ethernet_Data       (lBuffer, &lPacketInfo));
    KMS_TEST_ASSERT(                                               6 == Ethernet_DataSize   (lBuffer, &lPacketInfo));
    KMS_TEST_ASSERT(reinterpret_cast<unsigned short *>(lBuffer +  0) == Ethernet_Destination(lBuffer, &lPacketInfo));
    KMS_TEST_ASSERT(reinterpret_cast<unsigned short *>(lBuffer +  6) == Ethernet_Source     (lBuffer, &lPacketInfo));

    // ===== IPv4.h =========================================================

    KMS_TEST_COMPARE(752, IPv4_DataSize(lBuffer));
    KMS_TEST_COMPARE(0xa, IPv4_Protocol(lBuffer));

    KMS_TEST_ASSERT(                                  (lBuffer + 20) == IPv4_Data       (lBuffer));
    KMS_TEST_ASSERT(reinterpret_cast<unsigned short *>(lBuffer + 16) == IPv4_Destination(lBuffer));
    KMS_TEST_ASSERT(reinterpret_cast<unsigned short *>(lBuffer + 12) == IPv4_Source     (lBuffer));

    // ===== IPv6.h =========================================================

    KMS_TEST_COMPARE(1286, IPv6_DataSize(lBuffer));
    KMS_TEST_COMPARE( 0x7, IPv6_Protocol(lBuffer));

    KMS_TEST_ASSERT(                                  (lBuffer + 40) == IPv6_Data       (lBuffer));
    KMS_TEST_ASSERT(reinterpret_cast<unsigned short *>(lBuffer + 24) == IPv6_Destination(lBuffer));
    KMS_TEST_ASSERT(reinterpret_cast<unsigned short *>(lBuffer +  8) == IPv6_Source     (lBuffer));

    // ===== RegEx.h ========================================================

    RegEx lRE0;
    unsigned char lCounters[16];

    // a
    const RegEx_State lRES00[] =
    {
        REG_EX_STATE('a', 1, 1),
        REG_EX_STATE_OK        ,
    };

    REG_EX_CREATE(&lRE0, lRES00, lCounters);

    KMS_TEST_COMPARE(0, RegEx_Execute(&lRE0, 'b'));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "ax"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "a" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "a" ));

    // a+b
    const RegEx_State lRES01[] =
    {
        REG_EX_STATE('a', 1, 255),
        REG_EX_STATE('b', 1,   1),
        REG_EX_STATE_OK          ,
    };

    REG_EX_CREATE(&lRE0, lRES01, lCounters);

    KMS_TEST_COMPARE(3, RegEx_TestString    (&lRE0, "aabx"));
    KMS_TEST_COMPARE(3, RegEx_TestString_0  (&lRE0, "aab" ));
    KMS_TEST_COMPARE(3, RegEx_TestString_End(&lRE0, "aab" ));

    // \d
    const RegEx_State lRES02[] =
    {
        REG_EX_STATE_DIGIT(1, 1),
        REG_EX_STATE_OK         ,
    };

    REG_EX_CREATE(&lRE0, lRES02, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(0, RegEx_Execute(&lRE0, 'a'));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "0x"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "0" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "0" ));

    // \D
    const RegEx_State lRES03[] =
    {
        REG_EX_STATE_DIGIT_NOT(1, 1),
        REG_EX_STATE_OK             ,
    };

    REG_EX_CREATE(&lRE0, lRES03, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(0, RegEx_Execute(&lRE0, '0'));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "ax"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "a" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "a" ));

    // ^a
    const RegEx_State lRES04[] =
    {
        REG_EX_STATE_START     ,
        REG_EX_STATE('a', 1, 1),
        REG_EX_STATE_OK        ,
    };

    REG_EX_CREATE(&lRE0, lRES04, lCounters);

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "ax"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "a" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "a" ));

    // (a)
    const RegEx_State lRES05[] =
    {
        REG_EX_STATE_GROUP(1, 1, 2), // 0 ---+ <--+
        REG_EX_STATE_OK            , //      |    |
        REG_EX_STATE('a', 1, 1)    , // 2 <--+    |
        REG_EX_STATE_RETURN(0)     , // ----------+
    };

    REG_EX_CREATE(&lRE0, lRES05, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "ax"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "a" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "a" ));

    // (a)+b
    const RegEx_State lRES06[] =
    {
        REG_EX_STATE_GROUP(1, 255, 3), // 0 ---+ <--+
        REG_EX_STATE('b', 1, 1)      , //      |    |
        REG_EX_STATE_OK              , //      |    |
        REG_EX_STATE('a', 1, 1)      , // 3 <--+    |
        REG_EX_STATE_RETURN(0)       , // ----------+
    };

    REG_EX_CREATE(&lRE0, lRES06, lCounters);

    KMS_TEST_COMPARE(3, RegEx_TestString    (&lRE0, "aabx"));
    KMS_TEST_COMPARE(3, RegEx_TestString_0  (&lRE0, "aab" ));
    KMS_TEST_COMPARE(3, RegEx_TestString_End(&lRE0, "aab" ));

    // a|b
    const RegEx_State lRES07[] =
    {
        REG_EX_STATE_OR(1, 1, 2), // 0 ---+ <--+
        REG_EX_STATE_OK         , //      |    |
        REG_EX_STATE('a', 1, 1) , // 2 <--+    |
        REG_EX_STATE_RETURN(0)  , // ----------+
        REG_EX_STATE('b', 1, 1) , //           |
        REG_EX_STATE_RETURN(0)  , // ----------+
        REG_EX_STATE_OR_END     ,
    };

    REG_EX_CREATE(&lRE0, lRES07, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "ax"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "a" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "a" ));

    const RegEx_State lRES08[] =
    {
        REG_EX_STATE_OR_FAST(1, 1, 2), // -----+
        REG_EX_STATE_OK              , //      |
        REG_EX_STATE('a', 1, 1)      , // 2 <--+
        REG_EX_STATE('b', 1, 1)      ,
        REG_EX_STATE_OR_END          ,
    };

    REG_EX_CREATE(&lRE0, lRES08, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "ax"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "a" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "a" ));

    // (a|b)*c
    const RegEx_State lRES09[] =
    {
        REG_EX_STATE_OR(0, 255, 3), // 0 ---+ <--+
        REG_EX_STATE('c', 1, 1)   , //      |    |
        REG_EX_STATE_OK           , //      |    |
        REG_EX_STATE('a', 1, 1)   , // 3 <--+    |
        REG_EX_STATE_RETURN(0)    , // ----------+
        REG_EX_STATE( 'b', 1, 1)  , //           |
        REG_EX_STATE_RETURN(0)    , // ----------+
        REG_EX_STATE_OR_END       ,
    };

    REG_EX_CREATE(&lRE0, lRES09, lCounters);

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "cx" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "c"  ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "c"  ));
    KMS_TEST_COMPARE(2, RegEx_TestString_End(&lRE0, "ac" ));
    KMS_TEST_COMPARE(2, RegEx_TestString_End(&lRE0, "bc" ));
    KMS_TEST_COMPARE(3, RegEx_TestString_End(&lRE0, "abc"));

    const RegEx_State lRES10[] =
    {
        REG_EX_STATE_OR_FAST(0, 255, 3), // 0 ---+
        REG_EX_STATE('c', 1, 1)        , //      |
        REG_EX_STATE_OK                , //      |
        REG_EX_STATE('a', 1, 1)        , // 3 <--+
        REG_EX_STATE('b', 1, 1)        ,
        REG_EX_STATE_OR_END            ,
    };

    REG_EX_CREATE(&lRE0, lRES10, lCounters);

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "cx" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "c"  ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "c"  ));
    KMS_TEST_COMPARE(2, RegEx_TestString_End(&lRE0, "ac" ));
    KMS_TEST_COMPARE(2, RegEx_TestString_End(&lRE0, "bc" ));
    KMS_TEST_COMPARE(3, RegEx_TestString_End(&lRE0, "abc"));

    const RegEx_State lRES11[] =
    {
        REG_EX_STATE_GROUP(0, 255, 3), // 0 ---+ <--+
        REG_EX_STATE('c', 1, 1)      , //      |    |
        REG_EX_STATE_OK              , //      |    |
        { 'a', 1, 1, REG_EX_FLAG_OR }, // 3 <--+    |
        REG_EX_STATE('b', 1, 1)      , //           |
        REG_EX_STATE_RETURN(0)       , // ----------+
    };

    REG_EX_CREATE(&lRE0, lRES11, lCounters);

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "cx" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "c"  ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "c"  ));
    KMS_TEST_COMPARE(2, RegEx_TestString_End(&lRE0, "ac" ));
    KMS_TEST_COMPARE(2, RegEx_TestString_End(&lRE0, "bc" ));
    KMS_TEST_COMPARE(3, RegEx_TestString_End(&lRE0, "abc"));

    // [^a]
    const RegEx_State lRES12[] =
    {
        REG_EX_STATE_OR_NOT(1, 1, 2), // -----+
        REG_EX_STATE_OK             , //      |
        REG_EX_STATE('a', 1, 1)     , // 2 <--+
        REG_EX_STATE_OR_END         ,
    };

    REG_EX_CREATE(&lRE0, lRES12, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(0, RegEx_Execute(&lRE0, 'a'));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "cx"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "c" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "c" ));

    // [^a-c]
    const RegEx_State lRES13[] =
    {
        REG_EX_STATE_OR_NOT(1, 1, 2), // -----+
        REG_EX_STATE_OK             , //      |
        REG_EX_STATE_RANGE('a', 'c'), // 2 <--+
        REG_EX_STATE_OR_END         ,
    };

    REG_EX_CREATE(&lRE0, lRES13, lCounters);

    KMS_TEST_COMPARE(0, RegEx_Execute(&lRE0, 'a'));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "dx"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "d" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "d" ));

    // \s
    const RegEx_State lRES14[] =
    {
        REG_EX_STATE_SPACE(1, 1),
        REG_EX_STATE_OK         ,
    };

    REG_EX_CREATE(&lRE0, lRES14, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(0, RegEx_Execute(&lRE0, 'a'));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, " x"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, " " ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, " " ));

    // \S
    const RegEx_State lRES15[] =
    {
        REG_EX_STATE_SPACE_NOT(1, 1),
        REG_EX_STATE_OK             ,
    };

    REG_EX_CREATE(&lRE0, lRES15, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(0, RegEx_Execute(&lRE0, ' '));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "ax"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "a" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "a" ));

    // \w
    const RegEx_State lRES16[] =
    {
        REG_EX_STATE_WORD(1, 1),
        REG_EX_STATE_OK        ,
    };

    REG_EX_CREATE(&lRE0, lRES16, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(0, RegEx_Execute(&lRE0, ' '));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, "ax"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, "a" ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, "a" ));

    // \W
    const RegEx_State lRES17[] =
    {
        REG_EX_STATE_WORD_NOT(1, 1),
        REG_EX_STATE_OK            ,
    };

    REG_EX_CREATE(&lRE0, lRES17, lCounters);

    KMS_TEST_COMPARE(0, RegEx_End(&lRE0));

    KMS_TEST_COMPARE(0, RegEx_Execute(&lRE0, 'a'));

    KMS_TEST_COMPARE(1, RegEx_TestString    (&lRE0, " x"));
    KMS_TEST_COMPARE(1, RegEx_TestString_0  (&lRE0, " " ));
    KMS_TEST_COMPARE(1, RegEx_TestString_End(&lRE0, " " ));

    // a$
    const RegEx_State lRES18[] =
    {
        REG_EX_STATE('a', 1, 1),
        REG_EX_STATE_END       ,
    };

    REG_EX_CREATE(&lRE0, lRES18, lCounters);

    KMS_TEST_COMPARE(-1, RegEx_TestString    (&lRE0, "aa")); // ?

    KMS_TEST_COMPARE(-1, RegEx_TestString_0  (&lRE0, "ax"));
    KMS_TEST_COMPARE( 1, RegEx_TestString_0  (&lRE0, "a" ));
    KMS_TEST_COMPARE( 1, RegEx_TestString_End(&lRE0, "a" ));

    // \d{3}-\d{3}-\d{4}
    const RegEx_State lRES19[] =
    {
        REG_EX_STATE_DIGIT(3, 3),
        REG_EX_STATE('-', 1, 1) ,
        REG_EX_STATE_DIGIT(3, 3),
        REG_EX_STATE('-', 1, 1) ,
        REG_EX_STATE_DIGIT(4, 4),
        REG_EX_STATE_OK         ,
    };

    REG_EX_CREATE(&lRE0, lRES19, lCounters);

    KMS_TEST_COMPARE(12, RegEx_TestString    (&lRE0, "418-832-1208x"));
    KMS_TEST_COMPARE(12, RegEx_TestString_0  (&lRE0, "418-832-1208" ));
    KMS_TEST_COMPARE(12, RegEx_TestString_End(&lRE0, "418-832-1208" ));

    // \w+@[-a-z]+\.[a-z]{2,3}
    const RegEx_State lRES20[] =
    {
        REG_EX_STATE_WORD(1, 255)   ,
        REG_EX_STATE('@', 1, 1)     ,
        REG_EX_STATE_OR(1, 255, 6)  , //  2 ---+ <--+
        REG_EX_STATE('.', 1, 1)     , //       |    |
        REG_EX_STATE_OR(2, 3, 11)   , //  4 ---|-+  | <--+
        REG_EX_STATE_OK             , //       | |  |    |
        REG_EX_STATE('-', 1, 1)     , //  6 <--+ |  |    |
        REG_EX_STATE_RETURN(2)      , //  -------|--+    |
        REG_EX_STATE_RANGE('a', 'z'), //         |  |    |
        REG_EX_STATE_RETURN(2)      , //  -------|--+    |
        REG_EX_STATE_OR_END         , //         |       |
        REG_EX_STATE_RANGE('a', 'z'), // 11 <----+       |
        REG_EX_STATE_RETURN(4)      , // ----------------+
        REG_EX_OR_END               ,
    };

    REG_EX_CREATE(&lRE0, lRES20, lCounters);

    KMS_TEST_COMPARE(21, RegEx_TestString    (&lRE0, "mdubois@kms-quebec.com"));
    KMS_TEST_COMPARE(21, RegEx_TestString_0  (&lRE0, "mdubois@kms-quebec.com"));
    KMS_TEST_COMPARE(21, RegEx_TestString_End(&lRE0, "mdubois@kms-quebec.com"));

    const RegEx_State lRES21[] =
    {
        REG_EX_STATE_WORD(1, 255)      ,
        REG_EX_STATE('@', 1, 1)        ,
        REG_EX_STATE_OR_FAST(1, 255, 6), //  ----+
        REG_EX_STATE('.', 1, 1)        , //      |
        REG_EX_STATE_OR_FAST(2, 3, 9)  , //  ----|-+
        REG_EX_STATE_OK                , //      | |
        REG_EX_STATE('-', 1, 1)        , // 6 <--+ |
        REG_EX_STATE_RANGE('a', 'z')   , //        |
        REG_EX_STATE_OR_END            , //        |
        REG_EX_STATE_RANGE('a', 'z')   , // 9 <----+
        REG_EX_OR_END                  ,
    };

    REG_EX_CREATE(&lRE0, lRES21, lCounters);

    KMS_TEST_COMPARE(22, RegEx_TestString    (&lRE0, "mdubois@kms-quebec.comx"));
    KMS_TEST_COMPARE(22, RegEx_TestString_0  (&lRE0, "mdubois@kms-quebec.com" ));
    KMS_TEST_COMPARE(22, RegEx_TestString_End(&lRE0, "mdubois@kms-quebec.com" ));

    // .a
    const RegEx_State lRES22[] =
    {
        REG_EX_STATE_DOT(1, 1) ,
        REG_EX_STATE('a', 1, 1),
        REG_EX_STATE_OK        ,
    };

    REG_EX_CREATE(&lRE0, lRES22, lCounters);

    KMS_TEST_COMPARE(2, RegEx_TestString    (&lRE0, "aax"));
    KMS_TEST_COMPARE(2, RegEx_TestString_0  (&lRE0, "aa" ));
    KMS_TEST_COMPARE(2, RegEx_TestString_End(&lRE0, "aa" ));

    // ===== TCP.h ==========================================================

    KMS_TEST_COMPARE(0x0403, TCP_DestinationPort(lBuffer));
    KMS_TEST_COMPARE(0x0245, TCP_SourcePort     (lBuffer));

    KMS_TEST_ASSERT((lBuffer + 20) == TCP_Data(lBuffer));

    // ===== UDP.h ==========================================================

    KMS_TEST_COMPARE(0x0403, UDP_DestinationPort(lBuffer));
    KMS_TEST_COMPARE(0x0245, UDP_SourcePort     (lBuffer));

    KMS_TEST_ASSERT((lBuffer + 8) == UDP_Data(lBuffer));
}
KMS_TEST_END

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

int RegEx_TestString(RegEx * aRegEx, const char * aStr)
{
    assert(NULL != aRegEx);
    assert(NULL != aStr  );

    unsigned int i = 0;
    while ('\0' != aStr[i])
    {
        if (RegEx_Execute(aRegEx, aStr[i]))
        {
            return i;
        }

        i++;
    }

    return -1;
}

int RegEx_TestString_0(RegEx * aRegEx, const char * aStr)
{
    assert(NULL != aRegEx);
    assert(NULL != aStr  );

    unsigned int i = 0;
    while ('\0' != aStr[i])
    {
        if (RegEx_Execute(aRegEx, aStr[i]))
        {
            return i;
        }

        i++;
    }

    if (RegEx_Execute(aRegEx, '\0'))
    {
        return i;
    }

    return -1;
}

int RegEx_TestString_End(RegEx * aRegEx, const char * aStr)
{
    assert(NULL != aRegEx);
    assert(NULL != aStr  );

    unsigned int i = 0;
    while ('\0' != aStr[i])
    {
        if (RegEx_Execute(aRegEx, aStr[i]))
        {
            return i;
        }

        i++;
    }

    if (RegEx_End(aRegEx))
    {
        return i;
    }

    return -1;
}
