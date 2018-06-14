
// Author   KMS - Martin Dubois, ing.
// Product  OpenNet
// File     OpenNet/Status.cpp

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes/OpenNet ===================================================
#include <OpenNet/Status.h>

// Data type
/////////////////////////////////////////////////////////////////////////////

typedef struct
{
    const char * mName       ;
    const char * mDescription;
}
StatusInfo;

// Constants
/////////////////////////////////////////////////////////////////////////////

static const StatusInfo STATUS_INFO[] =
{
    { "OK", "Success" },

    { "CANNOT_OPEN_INPUT_FILE"   , "Cannot open input file"    },
    { "CANNOT_READ_INPUT_FILE"   , "Cannor read input file"    },
    { "CODE_ALREADY_SET"         , "Code already set"          },
    { "CODE_NOT_SET"             , "Code not set"              },
    { "CORRUPTED_DRIVER_DATA"    , "Corrupted driver data"     },
    { "DESTINATION_ALREADY_SET"  , "Destination already set"   },
    { "DESTINATION_NOT_SET"      , "Destination not set"       },
    { "EMPTY_CODE"               , "EMPTY_CODE"                },
    { "EMPTY_INPUT_FILE"         , "Empty input file"          },
    { "ERROR_CLOSING_FILE"       , "Error closing file"        },
    { "ERROR_READING_INPUT_FILE" , "Error reading input file"  },
    { "EXCEPTION"                , "Exception"                 },
    { "FILTER_ALREADY_SET"       , "Filter already set"        },
    { "FILTER_NOT_SET"           , "Filter not set"            },
    { "INPUT_FILE_TOO_LARGE"     , "Input file too large"      },
    { "INVALID_BUFFER_COUNT"     , "Invalid buffer count"      },
    { "INVALID_PROCESSOR"        , "Invalid processor"         },
    { "INVALID_REFERENCE"        , "Invalid reference"         },
    { "IOCTL_ERROR"              , "IoCtl error"               },
    { "NO_DESTINATION_SET"       , "No destination set"        },
    { "NOT_ALLOWER_NULL_ARGUMENT", "Not allower NULL argument" },
    { "NOT_CONNECTED"            , "Not connected"             },
    { "PACKET_TOO_LARGE"         , "Packet too large"          },
    { "PACKET_TOO_SMALL"         , "Packet too small"          },
    { "PROCESSOR_ALREADY_SET"    , "Processor already set"     },
    { "PROCESSOR_NOT_SET"        , "Processor not set"         },
};

namespace OpenNet
{

    // Functions
    /////////////////////////////////////////////////////////////////////////

    const char * Status_GetDescription(Status aStatus)
    {
        if (STATUS_QTY <= aStatus)
        {
            return "Invalid status code";
        }

        return STATUS_INFO[aStatus].mDescription;
    }

    const char * Status_GetName(Status aStatus)
    {
        if (STATUS_QTY <= aStatus)
        {
            return "INVALID_STATUS_CODE";
        }

        return STATUS_INFO[aStatus].mName;
    }

}
