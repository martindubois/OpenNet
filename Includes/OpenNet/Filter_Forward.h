
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Filter_Forward.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/Filter.h>

namespace OpenNet
{
    class Adapter;

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class define a filter simply forwarding packet.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe definit un filtre qui ne fait que transmettre
    ///         les paquets.
    /// \endcond
    class Filter_Forward : public Filter
    {

    public:

        /// \cond en
        /// \brief  Constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \endcond
        OPEN_NET_PUBLIC Filter_Forward();

        /// \cond en
        /// \brief  Add a destination adapter
        /// \param  aAdapter [---;RW-] The Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter un Adapter de destination
        /// \param  aAdapter [---;RW-] L'Adapter
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_DESTINATION_ALREADY_SET
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        OPEN_NET_PUBLIC Status AddDestination(Adapter * aAdapter);

        /// \cond en
        /// \brief  Remove a destination adapter
        /// \param  aAdapter [---;RW-] The Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Retirer un Adapter de destination
        /// \param  aAdapter [---;RW-] L'Adapter
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_DESTINATION_NOT_SET
        /// \retval STATUS_NO_DESTINATION_SET
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        OPEN_NET_PUBLIC Status RemoveDestination(Adapter * aAdapter);

        /// \cond en
        /// \brief  Remove all destination Adapter
        /// \endcond
        /// \cond fr
        /// \brief  Retirer tous les Adapter de destination
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NO_DESTINATION_NOT
        OPEN_NET_PUBLIC Status ResetDestinations();

        // ===== Filter =====================================================

        /// \cond en
        /// \brief  Destructor
        /// \endcond
        /// \cond fr
        /// \brief  Destructeur
        /// \endcond
        virtual OPEN_NET_PUBLIC ~Filter_Forward();

    private:

        Filter_Forward(const Filter_Forward &);

        const Filter_Forward & operator = (const Filter_Forward &);

        void GenerateCode();

        uint32_t mDestinations;

    };

}
