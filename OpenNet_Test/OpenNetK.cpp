
// Author     KMS - Martin Dubois, ing.
// Copyright  (C) 2018-2019 KMS. All rights reserved.
// Product    OpenNet
// File       OpenNet_Test/OpenNetK.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

#include "Component.h"

// ===== Includes ===========================================================

#define OPEN_NET_CONSTANT const
#define OPEN_NET_GLOBAL

#include <OpenNetK/Types.h>

#include <OpenNetK/ARP.h>
#include <OpenNetK/ByteOrder.h>
#include <OpenNetK/Ethernet.h>
#include <OpenNetK/IPv4.h>
#include <OpenNetK/IPv6.h>
#include <OpenNetK/RegEx.h>
#include <OpenNetK/TCP.h>
#include <OpenNetK/UDP.h>

// Data types
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    int          mExpectedResult;
    int       (* mFunction)(RegEx *, const char *);
    const char * mText;
}
RegEx_Test;

typedef struct
{
    const char * mRegEx;

    const RegEx_State mStates[14];

    const RegEx_Test mTests[7];
}
RegEx_Case;

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

static bool RegEx_Case_Run(const RegEx_Case * aCase, unsigned int aNo);

static int RegEx_TestChar      (RegEx * aRegEx, const char * aStr);
static int RegEx_TestEnd       (RegEx * aRegEx, const char * aStr);
static int RegEx_TestString    (RegEx * aRegEx, const char * aStr);
static int RegEx_TestString_0  (RegEx * aRegEx, const char * aStr);
static int RegEx_TestString_End(RegEx * aRegEx, const char * aStr);

// RegEx test list
/////////////////////////////////////////////////////////////////////////////

const RegEx_Case REG_EX_CASES[] =
{
    {
        "a",
        {
            REG_EX_STATE('a', 1, 1),
            REG_EX_STATE_OK
        },
        {
            { 0, RegEx_TestChar      , "b"  },
            { 1, RegEx_TestString    , "ax" },
            { 1, RegEx_TestString_0  , "a"  },
            { 1, RegEx_TestString_End, "a"  },
            { 0, NULL, NULL }
        }
    },

    {
        "a+b",
        {
            REG_EX_STATE('a', 1, 255),
            REG_EX_STATE('b', 1,   1),
            REG_EX_STATE_OK          ,
        },
        {
            { 3, RegEx_TestString    , "aabx" },
            { 3, RegEx_TestString_0  , "aab"  },
            { 3, RegEx_TestString_End, "aab"  },
            { 0, NULL, NULL }
        }
    },

    {
        "\\d",
        {
            REG_EX_STATE_DIGIT(1, 1),
            REG_EX_STATE_OK         ,
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 0, RegEx_TestChar      , "a"  },
            { 1, RegEx_TestString    , "0x" },
            { 1, RegEx_TestString_0  , "0"  },
            { 1, RegEx_TestString_End, "0"  },
            { 0, NULL, NULL }
        }
    },

    {
        "\\D",
        {
            REG_EX_STATE_DIGIT_NOT(1, 1),
            REG_EX_STATE_OK             ,
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 0, RegEx_TestChar      , "0"  },
            { 1, RegEx_TestString    , "ax" },
            { 1, RegEx_TestString_0  , "a"  },
            { 1, RegEx_TestString_End, "a"  },
            { 0, NULL, NULL }
        }
    },

    {
        "^a",
        {
            REG_EX_STATE_START     ,
            REG_EX_STATE('a', 1, 1),
            REG_EX_STATE_OK        ,
        },
        {
            { 1, RegEx_TestString    , "ax" },
            { 1, RegEx_TestString_0  , "a"  },
            { 1, RegEx_TestString_End, "a"  },
            { 0, NULL, NULL }
        }
    },

    {
        "(a)",
        {
            REG_EX_STATE_GROUP(1, 1, 2), // 0 ---+ <--+
            REG_EX_STATE_OK            , //      |    |
            REG_EX_STATE('a',  1, 1)   , // 2 <--+    |
            REG_EX_STATE_RETURN(     0), // ----------+
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 1, RegEx_TestString    , "ax" },
            { 1, RegEx_TestString_0  , "a"  },
            { 1, RegEx_TestString_End, "a"  },
            { 0, NULL, NULL }
        }
    },

    {
        "(a)+b",
        {
            REG_EX_STATE_GROUP(1, 255, 3), // 0 ---+ <--+
            REG_EX_STATE('b',  1,   1)   , //      |    |
            REG_EX_STATE_OK              , //      |    |
            REG_EX_STATE('a',  1,   1)   , // 3 <--+    |
            REG_EX_STATE_RETURN(       0), // ----------+
        },
        {
            { 3, RegEx_TestString    , "aabx" },
            { 3, RegEx_TestString_0  , "aab"  },
            { 3, RegEx_TestString_End, "aab"  },
            { 0, NULL, NULL }
        }
    },

    {
        "a|b",
        {
            REG_EX_STATE_OR(  1, 1, 2), // 0 ---+ <--+
            REG_EX_STATE_OK           , //      |    |
            REG_EX_STATE('a', 1, 1)   , // 2 <--+    |
            REG_EX_STATE_RETURN(    0), // ----------+
            REG_EX_STATE('b', 1, 1)   , //           |
            REG_EX_STATE_RETURN(    0), // ----------+
            REG_EX_STATE_OR_END       ,
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 1, RegEx_TestString    , "ax" },
            { 1, RegEx_TestString_0  , "a"  },
            { 1, RegEx_TestString_End, "a"  },
            { 0, NULL, NULL }
        }
    },

    {
        "[ab]",
        {
            REG_EX_STATE_OR_FAST(1, 1, 2), // -----+
            REG_EX_STATE_OK              , //      |
            REG_EX_STATE('a',    1, 1)   , // 2 <--+
            REG_EX_STATE('b',    1, 1)   ,
            REG_EX_STATE_OR_END          ,
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 1, RegEx_TestString    , "ax" },
            { 1, RegEx_TestString_0  , "a"  },
            { 1, RegEx_TestString_End, "a"  },
            { 0, NULL, NULL }
        }
    },

    {
        "(a|b)*c",
        {
            REG_EX_STATE_OR(  0, 255, 3), // 0 ---+ <--+
            REG_EX_STATE('c', 1, 1)     , //      |    |
            REG_EX_STATE_OK             , //      |    |
            REG_EX_STATE('a', 1, 1)     , // 3 <--+    |
            REG_EX_STATE_RETURN(      0), // ----------+
            REG_EX_STATE('b', 1, 1)     , //           |
            REG_EX_STATE_RETURN(      0), // ----------+
            REG_EX_STATE_OR_END         ,
        },
        {
            { 1, RegEx_TestString    , "cx"  },
            { 1, RegEx_TestString_0  , "c"   },
            { 1, RegEx_TestString_End, "c"   },
            { 2, RegEx_TestString_End, "ac"  },
            { 2, RegEx_TestString_End, "bc"  },
            { 3, RegEx_TestString_End, "abc" },
            { 0, NULL, NULL }
        }
    },

    {
        "[ab]*c",
        {
            REG_EX_STATE_OR_FAST(0, 255, 3), // -----+
            REG_EX_STATE('c',    1,   1)   , //      |
            REG_EX_STATE_OK                , //      |
            REG_EX_STATE('a',    1,   1)   , // 3 <--+
            REG_EX_STATE('b',    1,   1)   ,
            REG_EX_STATE_OR_END            ,
        },
        {
            { 1, RegEx_TestString    , "cx"  },
            { 1, RegEx_TestString_0  , "c"   },
            { 1, RegEx_TestString_End, "c"   },
            { 2, RegEx_TestString_End, "ac"  },
            { 2, RegEx_TestString_End, "bc"  },
            { 3, RegEx_TestString_End, "abc" },
            { 0, NULL, NULL }
        }
    },

    {
        "([ab])*c",
        {
            REG_EX_STATE_GROUP(0, 255, 3), // 0 ---+ <--+
            REG_EX_STATE('c',  1,   1)   , //      |    |
            REG_EX_STATE_OK              , //      |    |
            { 'a', 1, 1, REG_EX_FLAG_OR }, // 3 <--+    |
            REG_EX_STATE('b',  1,   1)   , //           |
            REG_EX_STATE_RETURN(       0), // ----------+
        },
        {
            { 1, RegEx_TestString    , "cx"  },
            { 1, RegEx_TestString_0  , "c"   },
            { 1, RegEx_TestString_End, "c"   },
            { 2, RegEx_TestString_End, "ac"  },
            { 2, RegEx_TestString_End, "bc"  },
            { 3, RegEx_TestString_End, "abc" },
            { 0, NULL, NULL }
        }
    },

    {
        "[^a]",
        {
            REG_EX_STATE_OR_NOT(1, 1, 2), // -----+
            REG_EX_STATE_OK             , //      |
            REG_EX_STATE('a',   1, 1)   , // 2 <--+
            REG_EX_STATE_OR_END         ,
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 0, RegEx_TestChar      , "a"  },
            { 1, RegEx_TestString    , "cx" },
            { 1, RegEx_TestString_0  , "c"  },
            { 1, RegEx_TestString_End, "c"  },
            { 0, NULL, NULL }
        }
    },

    {
        "[^a-c]",
        {
            REG_EX_STATE_OR_NOT(1, 1, 2), // -----+
            REG_EX_STATE_OK             , //      |
            REG_EX_STATE_RANGE('a', 'c'), // 2 <--+
            REG_EX_STATE_OR_END         ,
        },
        {
            { 0, RegEx_TestChar      , "a"  },
            { 1, RegEx_TestString    , "dx" },
            { 1, RegEx_TestString_0  , "d"  },
            { 1, RegEx_TestString_End, "d"  },
            { 0, NULL, NULL }
        }
    },

    {
        "\\s",
        {
            REG_EX_STATE_SPACE(1, 1),
            REG_EX_STATE_OK         ,
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 0, RegEx_TestChar      , "a"  },
            { 1, RegEx_TestString    , " x" },
            { 1, RegEx_TestString_0  , " "  },
            { 1, RegEx_TestString_End, " "  },
            { 0, NULL, NULL }
        }
    },

    {
        "\\S",
        {
            REG_EX_STATE_SPACE_NOT(1, 1),
            REG_EX_STATE_OK             ,
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 0, RegEx_TestChar      , " "  },
            { 1, RegEx_TestString    , "ax" },
            { 1, RegEx_TestString_0  , "a"  },
            { 1, RegEx_TestString_End, "a"  },
            { 0, NULL, NULL }
        }
    },

    {
        "\\w",
        {
            REG_EX_STATE_WORD(1, 1),
            REG_EX_STATE_OK        ,
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 0, RegEx_TestChar      , " "  },
            { 1, RegEx_TestString    , "ax" },
            { 1, RegEx_TestString_0  , "a"  },
            { 1, RegEx_TestString_End, "a"  },
            { 0, NULL, NULL }
        }
    },

    {
        "\\W",
        {
            REG_EX_STATE_WORD_NOT(1, 1),
            REG_EX_STATE_OK            ,
        },
        {
            { 0, RegEx_TestEnd       , NULL },
            { 0, RegEx_TestChar      , "a"  },
            { 1, RegEx_TestString    , " x" },
            { 1, RegEx_TestString_0  , " "  },
            { 1, RegEx_TestString_End, " "  },
            { 0, NULL, NULL }
        }
    },

    {
        "a$",
        {
            REG_EX_STATE('a', 1, 1),
            REG_EX_STATE_END       ,
        },
        {
            { -1, RegEx_TestString    , "ax" },
            {  1, RegEx_TestString_0  , "a"  },
            {  1, RegEx_TestString_End, "a"  },
            {  2, RegEx_TestString_End, "aa" },
            { 0, NULL, NULL }
        }
    },

    // Phone number
    {
        "\\d{3}-\\d{3}-\\d{4}",
        {
            REG_EX_STATE_DIGIT(3, 3),
            REG_EX_STATE('-',  1, 1),
            REG_EX_STATE_DIGIT(3, 3),
            REG_EX_STATE('-',  1, 1),
            REG_EX_STATE_DIGIT(4, 4),
            REG_EX_STATE_OK         ,
        },
        {
            { 12, RegEx_TestString    , "418-832-1208x" },
            { 12, RegEx_TestString_0  , "418-832-1208"  },
            { 12, RegEx_TestString_End, "418-832-1208"  },
            { 0, NULL, NULL }
        }
    },

    // E-mail address where the domain name contains 2 parts
    {
        "\\w+@[-a-z]+\\.[a-z]{2,3}",
        {
            REG_EX_STATE_WORD(1, 255)    ,
            REG_EX_STATE('@', 1,   1)    ,
            REG_EX_STATE_OR(  1, 255,  6), //  2 ---+ <--+
            REG_EX_STATE('.', 1,   1)    , //       |    |
            REG_EX_STATE_OR(  2,   3, 11), //  4 ---|-+  | <--+
            REG_EX_STATE_OK              , //       | |  |    |
            REG_EX_STATE('-', 1,   1)    , //  6 <--+ |  |    |
            REG_EX_STATE_RETURN(       2), // --------|--+    |
            REG_EX_STATE_RANGE('a', 'z') , //         |  |    |
            REG_EX_STATE_RETURN(       2), // --------|--+    |
            REG_EX_STATE_OR_END          , //         |       |
            REG_EX_STATE_RANGE('a', 'z') , // 11 <----+       |
            REG_EX_STATE_RETURN(       4), // ----------------+
            REG_EX_OR_END                ,
        },
        {
            { 21, RegEx_TestString    , "mdubois@kms-quebec.com" },
            { 21, RegEx_TestString_0  , "mdubois@kms-quebec.com" },
            { 21, RegEx_TestString_End, "mdubois@kms-quebec.com" },
            { 0, NULL, NULL }
        }
    },

    // Second state list for the previous regular expression
    {
        "\\w+@[-a-z]+\\.[a-z]{2,3}",
        {
            REG_EX_STATE_WORD(   1, 255)   ,
            REG_EX_STATE('@',    1,   1)   ,
            REG_EX_STATE_OR_FAST(1, 255, 6), // -----+
            REG_EX_STATE('.',    1,   1)   , //      |
            REG_EX_STATE_OR_FAST(2,   3, 9), // -----|-+
            REG_EX_STATE_OK                , //      | |
            REG_EX_STATE('-',    1,   1)   , // 6 <--+ |
            REG_EX_STATE_RANGE('a', 'z')   , //        |
            REG_EX_STATE_OR_END            , //        |
            REG_EX_STATE_RANGE('a', 'z')   , // 9 <----+
            REG_EX_OR_END                  ,
        },
        {
            { 22, RegEx_TestString    , "mdubois@kms-quebec.comx" },
            { 22, RegEx_TestString_0  , "mdubois@kms-quebec.com"  },
            { 22, RegEx_TestString_End, "mdubois@kms-quebec.com"  },
            { 0, NULL, NULL }
        }
    },

    {
        ".a",
        {
            REG_EX_STATE_DOT( 1, 1),
            REG_EX_STATE('a', 1, 1),
            REG_EX_STATE_OK        ,
        },
        {
            { 2, RegEx_TestString    , "aax" },
            { 2, RegEx_TestString_0  , "aa"  },
            { 2, RegEx_TestString_End, "aa"  },
            { 0, NULL, NULL }
        }
    },

    // General e-mail address
    {
        "\\w+@([-a-z]+\\.)+[a-z]{2,4}",
        {
            REG_EX_STATE_WORD(   1, 255),
            REG_EX_STATE('@',    1,   1),
            REG_EX_STATE_GROUP(  1, 255,  5), //  2 ---+ <--+
            REG_EX_STATE_OR_FAST(2,   4,  8), // ------|----|--+
            REG_EX_STATE_OK                 , //       |    |  |    
            REG_EX_STATE_OR_FAST(1, 255, 10), //  5 <--+ ---|--|--+
            REG_EX_STATE('.',    1,   1)    , //            |  |  |
            REG_EX_STATE_RETURN(          2), // -----------+  |  |
            REG_EX_STATE_RANGE('a', 'z')    , //  8 <----------+  |
            REG_EX_STATE_OR_END             , //                  |
            REG_EX_STATE('-',    1,   1)    , // 10 <-------------+
            REG_EX_STATE_RANGE('a', 'z')    ,
            REG_EX_STATE_OR_END             ,
        },
        {
            { 22, RegEx_TestString    , "mdubois@kms-quebec.com " },
            { 22, RegEx_TestString_0  , "mdubois@kms-quebec.com"  },
            { 22, RegEx_TestString_End, "mdubois@kms-quebec.com"  },
            { 0, NULL, NULL }
        }
    },

    { NULL, {}, {} }
};

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

KMS_TEST_BEGIN(OpenNetK_RegEx)
{
    unsigned int i = 0;

    while (NULL != REG_EX_CASES[i].mRegEx)
    {
        KMS_TEST_ASSERT(RegEx_Case_Run(REG_EX_CASES + i, i));

        i++;
    }
}
KMS_TEST_END

// Static function declarations
/////////////////////////////////////////////////////////////////////////////

bool RegEx_Case_Run(const RegEx_Case * aCase, unsigned int aNo)
{
    assert(NULL != aCase);

    bool lResult = true;

    RegEx         lRegEx;
    unsigned char lCounters[32];

    REG_EX_CREATE(&lRegEx, aCase->mStates, lCounters);

    unsigned int i = 0;

    while (NULL != aCase->mTests[i].mFunction)
    {
        const RegEx_Test * lTest = aCase->mTests + i;

        int lRet = lTest->mFunction(&lRegEx, lTest->mText);

        if (lTest->mExpectedResult != lRet)
        {
            if (lResult)
            {
                printf("ERROR  Case %u  %s\n", aNo, aCase->mRegEx);
                printf("  Test Expected     Got      Text\n");
                printf("  ---- --------    ----- ------------\n");
                lResult = false;
            }

            printf("  %3u\t%6d  !=%5d  %s\n", i, lTest->mExpectedResult, lRet, lTest->mText);
        }

        i++;
    }

    return lResult;
}

int RegEx_TestChar(RegEx * aRegEx, const char * aStr)
{
    assert(NULL != aRegEx);
    assert(NULL != aStr  );

    return RegEx_Execute(aRegEx, aStr[0]);
}

int RegEx_TestEnd(RegEx * aRegEx, const char *)
{
    assert(NULL != aRegEx);

    return RegEx_End(aRegEx);
}

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
