
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

    { "ADAPTER_ALREADY_CONNECTED", "The adapter is already connected"},
    { "ADAPTER_NOT_CONNECTED"    , "The adapter is not connected"    },
    { "BUFFER_ALLOCATED"         , "At least one buffer is allocated"},
    { "BUFFER_TOO_SMALL"         , "The buffer is too small"         },
    { "CANNOT_OPEN_INPUT_FILE"   , "Cannot open input file"          },
    { "CANNOT_READ_INPUT_FILE"   , "Cannor read input file"          },
    { "CODE_ALREADY_SET"         , "The code is already set"         },
    { "CODE_NOT_SET"             , "The code is not set"             },
    { "CORRUPTED_DRIVER_DATA"    , "Corrupted driver data"           },
    { "DESTINATION_ALREADY_SET"  , "The destination is already set"  },
    { "DESTINATION_NOT_SET"      , "The destination is not set"      },
    { "EMPTY_CODE"               , "The code is empty"               },
    { "EMPTY_INPUT_FILE"         , "The input file is empty"         },
    { "ERROR_CLOSING_FILE"       , "Error closing file"              },
    { "ERROR_READING_INPUT_FILE" , "Error reading input file"        },
    { "EXCEPTION"                , "Exception"                       },
    { "FILTER_ALREADY_SET"       , "The filter is already set"       },
    { "FILTER_NOT_SET"           , "The filter is not set"           },
    { "FILTER_SET"               , "The filter is set"               },
    { "INPUT_FILE_TOO_LARGE"     , "The input file is too large"     },
    { "INTERNAL_ERROR"           , "Internal error"                  },
    { "INVALID_ADAPTER"          , "Invalid adapter"                 },
    { "INVALID_BUFFER_COUNT"     , "Invalid buffer count"            },
    { "INVALID_PACKET_SIZE"      , "Invalid packet size"             },
    { "INVALID_PROCESSOR"        , "Invalid processor"               },
    { "INVALID_REFERENCE"        , "Invalid reference"               },
    { "IOCTL_ERROR"              , "IoCtl error"                     },
    { "NO_ADAPTER_CONNECTED"     , "No adapter connected"            },
    { "NO_DESTINATION_SET"       , "No destination set"              },
    { "NOT_ALLOWED_NULL_ARGUMENT", "Not allowed NULL argument"       },
    { "OPEN_CL_ERROR"            , "OpenCL error"                    },
    { "PACKET_TOO_LARGE"         , "The packet is too large"         },
    { "PACKET_TOO_SMALL"         , "The packet is too small"         },
    { "PROCESSOR_ALREADY_SET"    , "The processor is already set"    },
    { "PROCESSOR_NOT_SET"        , "The processor is not set"        },
    { "SYSTEM_ALREADY_STARTED"   , "The system is already started"   },
    { "SYSTEM_NOT_STARTED"       , "The system is not started"       },
    { "TOO_MANY_BUFFER"          , "Too many buffer allocated"       },
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

    Status Status_Display(Status aStatus, FILE * aOut)
    {
        if (NULL == aOut)
        {
            return STATUS_NOT_ALLOWED_NULL_ARGUMENT;
        }

        fprintf(aOut, "%u = %s - %s\n", aStatus, Status_GetName(aStatus), Status_GetDescription(aStatus));

        return STATUS_OK;
    }

}