
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/Filter.h

#pragma once

// Includes
/////////////////////////////////////////////////////////////////////////////

// ===== Includes ===========================================================
#include <OpenNet/OpenNet.h>
#include <OpenNet/Status.h>

namespace OpenNet
{

    // Class
    /////////////////////////////////////////////////////////////////////////

    /// \cond en
    /// \brief  This class define the filter level interface.
    /// \endcond
    /// \cond fr
    /// \brief  Cette classe definit l'interface au niveau du filtre.
    /// \endcond
    class Filter
    {

    public:

        /// \cond en
        /// \brief  Constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \endcond
        OPEN_NET_PUBLIC Filter();

        /// \cond en
        /// \brief  Destructor
        /// \endcond
        /// \cond fr
        /// \brief  Destructeur
        /// \endcond
        virtual OPEN_NET_PUBLIC ~Filter();

        /// \cond en
        /// \brief  Retrieve the build log
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le log de compilation
        /// \retval Cette methode retourne l'adresse d'un espace de memoire
        ///         interne.
        /// \endcond
        OPEN_NET_PUBLIC const char * GetBuildLog() const;

        /// \cond en
        /// \brief  Retrieve the code line count
        /// \return This method returns the code line cout.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le nombre de ligne de code
        /// \retval Cette methode retourne le nombre de ligne de code.
        /// \endcond
        OPEN_NET_PUBLIC unsigned int  GetCodeLineCount();

        /// \cond en
        /// \brief  Retrieve the code lines
        /// \return This method returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir les lignes de code
        /// \retval Cette methode retourne l'adresse d'un espace de memoire
        ///         interne.
        /// \endcond
        OPEN_NET_PUBLIC const char ** GetCodeLines();

        /// \cond en
        /// \brief  Get code size
        /// \return The code size in byte
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir la taille du code
        /// \return La taille du code en octet
        /// \endcond
        OPEN_NET_PUBLIC unsigned int GetCodeSize() const;

        /// \cond en
        /// \brief  Retrieve the instance name
        /// \return This methode returns the address of an internal buffer.
        /// \endcond
        /// \cond fr
        /// \brief  Obtenir le nom de l'instance
        /// \return Cette methode retourne l'adresse d'un espace de memoire
        ///         interne.
        /// \endcond
        OPEN_NET_PUBLIC const char * GetName() const;

        /// \cond en
        /// \brief  Reset the code using a source file
        /// \endcond
        /// \cond fr
        /// \brief  Reset le code en utilisant un fichier source
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_CODE_NOT_SET
        OPEN_NET_PUBLIC Status ResetCode();

        /// \cond en
        /// \brief  Set the code using a source file
        /// \param  aFileName [---;R--] The source file name
        /// \endcond
        /// \cond fr
        /// \brief  Assigner le code en utilisant un fichier source
        /// \param  aFileName [---;R--] Le nom du fichier source
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_CANNOT_OPEN_INPUT_FILE
        /// \retval STATUS_CANNOT_READ_INPUT_FILE
        /// \retval STATUS_CODE_ALREADY_SET
        /// \retval STATUS_EMPTY_INPUT_FILE
        /// \retval STATUS_ERROR_CLOSING_INPUT_FILE
        /// \retval STATUS_ERROR_READING_INPUT_FILE
        /// \retval STATUS_INPUT_FILE_TOO_LARGE
        OPEN_NET_PUBLIC Status SetCode(const char * aFileName);

        /// \cond en
        /// \brief  Set the code
        /// \param  aCode [DK-;R--] The code
        /// \param  aCodeSize_byte  La taille du code
        /// \endcond
        /// \cond fr
        /// \brief  Assigner le code en utilisant un fichier source
        /// \param  aCode [DK-;R--] Le code
        /// \param  aCodeSize_byte  La taille du code
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_CODE_ALREADY_SET
        /// \retval STATUS_EMPTY_CODE
        OPEN_NET_PUBLIC Status SetCode(const char * aCode, unsigned int aCodeSize_byte);

        /// \cond en
        /// \brief  Set the instance's name
        /// \param  aName [---;R--] The name
        /// \endcond
        /// \cond fr
        /// \brief  Assigner le nom de l'instance
        /// \param  aName [---;R--] Le nom
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        OPEN_NET_PUBLIC Status SetName(const char * aName);

        /// \cond en
        /// \brief  Add the kernel arguments other than the first one
        /// \param  aKernel [---;R--] The cl_kernel instance
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter les argument du kernel autre que le premier
        /// \param  aKernel [---;R--] L'instance de cl_kernel
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWED_NULL_ARGUMENT
        virtual OPEN_NET_PUBLIC void AddKernelArgs(void * aKernal);

        /// \cond en
        /// \brief  Display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche
        /// \retval aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        OPEN_NET_PUBLIC Status Display(FILE * aOut) const;

        /// \cond en
        /// \brief  Replace strings in code
        /// \param  aSearch  [---;R--] The string to search for and remove
        /// \return This method returns the number of removed string.
        /// \endcond
        /// \cond fr
        /// \brief  Remplacer des chaines dans le code
        /// \param  aSearch  [---;R--] La chaine a chercher
        /// \return Cette methode retourne le nombre de chaine retirees.
        /// \endcond
        OPEN_NET_PUBLIC unsigned int Edit_Remove(const char * aSearch);

        /// \cond en
        /// \brief  Replace strings in code
        /// \param  aSearch  [---;R--] The string to search for
        /// \param  aReplace [---;R--] The string to use for replacing found
        ///                            strings
        /// \return This method returns the number of replacement.
        /// \endcond
        /// \cond fr
        /// \brief  Remplacer des chaines dans le code
        /// \param  aSearch  [---;R--] La chaine a chercher
        /// \param  aReplace [---;R--] La chaine a utiliser pour remplacer
        ///                            les chaines trouvees
        /// \return Cette methode retourne le nombre de remplacements
        ///         effectues
        /// \endcond
        OPEN_NET_PUBLIC unsigned int Edit_Replace(const char * aSearch, const char * aReplace);

        /// \cond en
        /// \brief  Search string in code
        /// \param  aSearch  [---;R--] The string to search for
        /// \return This method returns the number of fount occurence.
        /// \endcond
        /// \cond fr
        /// \brief  Rechercher une chaines dans le code
        /// \param  aSearch  [---;R--] La chaine a chercher
        /// \return Cette methode retourne le nombre d'instance trouvees.
        /// \endcond
        OPEN_NET_PUBLIC unsigned int Edit_Search(const char * aSearch);

    // internal

        static const unsigned int BUILD_LOG_MAX_SIZE_byte;

        void * AllocateBuildLog();

    private:

        Filter(const Filter &);

        const Filter & operator = (const Filter &);

        void CodeLines_Count   ();
        void CodeLines_Generate();

        unsigned int Edit_Replace_ByEqual  (const char * aSearch, const char * aReplace, unsigned int aLength);
        unsigned int Edit_Replace_ByLonger (const char * aSearch, const char * aReplace, unsigned int aSearchLength, unsigned int aReplaceLength);
        unsigned int Edit_Replace_ByShorter(const char * aSearch, const char * aReplace, unsigned int aSearchLength, unsigned int aReplaceLength);

        void Invalidate();

        void ReleaseCode();

        char        * mBuildLog      ;
        char        * mCode          ;
        char        * mCodeLineBuffer;
        unsigned int  mCodeLineCount ;
        const char ** mCodeLines     ;
        unsigned int  mCodeSize_byte ;
        char          mName[64]      ;

    };

}
