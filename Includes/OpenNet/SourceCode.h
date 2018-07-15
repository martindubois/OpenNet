
// Product  OpenNet

/// \author  KMS - Martin Dubois, ing.
/// \file    Includes/OpenNet/SourceCode.h

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
    /// \brief  The SourceCode class
    /// \endcond
    /// \cond fr
    /// \brief  La classe SourceCode
    /// \endcond
    class SourceCode
    {

    public:

        /// \cond en
        /// \brief  Constructor
        /// \endcond
        /// \cond fr
        /// \brief  Constructeur
        /// \endcond
        OPEN_NET_PUBLIC SourceCode();

        /// \cond en
        /// \brief  Destructor
        /// \endcond
        /// \cond fr
        /// \brief  Destructeur
        /// \endcond
        OPEN_NET_PUBLIC virtual ~SourceCode();

        /// \cond en
        /// \brief  Append code using a source file
        /// \param  aFileName [---;R--] The source file name
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter du code en utilisant un fichier source
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
        OPEN_NET_PUBLIC Status AppendCode(const char * aFileName);

        /// \cond en
        /// \brief  Appen code
        /// \param  aCode [---;R--] The code
        /// \param  aCodeSize_byte  La taille du code
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter du code
        /// \param  aCode [---;R--] Le code
        /// \param  aCodeSize_byte  La taille du code
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_CODE_ALREADY_SET
        /// \retval STATUS_EMPTY_CODE
        OPEN_NET_PUBLIC virtual Status AppendCode(const char * aCode, unsigned int aCodeSize_byte);

        /// \cond en
        /// \brief  Appen code
        /// \param  aCode [---;R--] The code
        /// \param  aCodeSize_byte  La taille du code
        /// \endcond
        /// \cond fr
        /// \brief  Ajouter du code
        /// \param  aCode [---;R--] Le code
        /// \param  aCodeSize_byte  La taille du code
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_CODE_ALREADY_SET
        /// \retval STATUS_EMPTY_CODE
        OPEN_NET_PUBLIC Status AppendCode(const SourceCode & aCode);

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
        OPEN_NET_PUBLIC virtual Status ResetCode();

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
        OPEN_NET_PUBLIC virtual Status SetCode(const char * aFileName);

        /// \cond en
        /// \brief  Set the code
        /// \param  aCode [---;R--] The code
        /// \param  aCodeSize_byte  La taille du code
        /// \endcond
        /// \cond fr
        /// \brief  Assigner le code
        /// \param  aCode [---;R--] Le code
        /// \param  aCodeSize_byte  La taille du code
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_CODE_ALREADY_SET
        /// \retval STATUS_EMPTY_CODE
        OPEN_NET_PUBLIC virtual Status SetCode(const char * aCode, unsigned int aCodeSize_byte);

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
        /// \brief  Display
        /// \param  aOut [---;RW-] The output stream
        /// \endcond
        /// \cond fr
        /// \brief  Affiche
        /// \retval aOut [---;RW-] Le fichier de sortie
        /// \endcond
        /// \retval STATUS_OK
        /// \retval STATUS_NOT_ALLOWER_NULL_ARGUMENT
        OPEN_NET_PUBLIC virtual Status Display(FILE * aOut) const;

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
        OPEN_NET_PUBLIC virtual unsigned int Edit_Remove(const char * aSearch);

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
        OPEN_NET_PUBLIC virtual unsigned int Edit_Replace(const char * aSearch, const char * aReplace);

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

    // internal:

        const char * GetCode() const;

    protected:

        char       * mCode         ;
        unsigned int mCodeSize_byte;

    private:

        SourceCode(const SourceCode &);

        const SourceCode & operator = (const SourceCode &);

        unsigned int Edit_Replace_ByEqual  (const char * aSearch, const char * aReplace, unsigned int aLength);
        unsigned int Edit_Replace_ByLonger (const char * aSearch, const char * aReplace, unsigned int aSearchLength, unsigned int aReplaceLength);
        unsigned int Edit_Replace_ByShorter(const char * aSearch, const char * aReplace, unsigned int aSearchLength, unsigned int aReplaceLength);

        void ReleaseCode();

        char mName[64];

    };

}
