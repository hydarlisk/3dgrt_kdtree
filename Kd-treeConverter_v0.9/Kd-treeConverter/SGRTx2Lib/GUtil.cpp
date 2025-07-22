#include "GUtil.h"
#include <sys/timeb.h>
#include <time.h>


GUtil::GUtil(void)
{
}

GUtil::~GUtil(void)
{
}

const char* GUtil::getFileExt( const char* szFileName )
{
	if ( szFileName == NULL )
		return NULL;

	int length = (int) strlen( szFileName );

	for ( int i = length - 1; i >= 0; --i ) {
		if ( szFileName[ i ] == '.' ) {
			return szFileName + i + 1;
		}
	}

	return NULL;
}

const char* GUtil::getParentPath( const char* szFileName, char* result )
{
	if ( szFileName == NULL )
		return NULL;

	result[ 0 ] = 0x00;

	int length = (int) strlen( szFileName );

	for ( int i = length - 1; i >= 0; --i ) {
		if ( szFileName[ i ] == FILE_SEPARATOR ) {
			strncpy( result, szFileName, i );
			result[ i ] = 0x00;
			return result;
		}
	}

	strcpy( result, szFileName );

	return result;

}

const char* GUtil::getFileName( const char* szFileName, char* result )
{
	if ( szFileName == NULL )
		return NULL;

	result[ 0 ] = 0x00;

	int length = (int) strlen( szFileName );
	int dotidx = length;

	for ( int i = length - 1; i >= 0; --i ) {
		if ( szFileName[ i ] == '.' ) {
			dotidx = i;
		} else if ( szFileName[ i ] == FILE_SEPARATOR ) {
			strncpy( result, szFileName + i + 1, dotidx - i - 1 );
			result[ dotidx - i - 1 ] = 0x00;
			return result;
		}
	}

	strcpy( result, szFileName );

	return result;

}

void GUtil::removeCRLF( char *data )
{
	int length = (int) strlen( data );

	if ( length >= 1 ) {
		if ( data[ length - 1 ] == '\n' || data[ length - 1 ] == '\r' )
			data[ length - 1 ] = 0x00;
	}

	if ( length >= 2 ) {
		if ( data[ length - 2 ] == '\n' || data[ length - 2 ] == '\r' )
			data[ length - 2 ] = 0x00;
	}

	length = (int) strlen( data );
	for (int i = length - 1; i >= 0; i--) {
		if (data[ i ] == 0x00) continue;
		if (data[ i ] == ' ' || data[ i ] == '\t') data[ i ] = 0x00;
		else break;
	}

}

void GUtil::getKeyValue( char *data, char **key, char **value )
{
	int length = (int) strlen( data );
	
	(*key) = data;
	(*value) = data;

	for ( int i = 0; i < length; ++i ) {
		if ( data[ i ] == '=' ) {
			data[ i ] = 0x00;
			(*value) = ( data + i + 1 );
			return;
		}
	}
}

/**
 *	주의: strtok 함수를 사용하므로 thread-safe 하지 않다.
 */
void GUtil::getValueList( char *data, vector<char*> &valueList )
{
	return getValueList( data, valueList, " \t" );
}

void GUtil::getValueList( char *data, vector<char*> &valueList, const char* delimeter )
{
	valueList.clear();

	char *value = strtok( data, delimeter );
	while( value != NULL ) {
		valueList.push_back( value );
		value = strtok( NULL, delimeter );
	}
}

char* GUtil::copyString( const char *src )
{
	if ( src == NULL )
		return NULL;

	char *newStr = (char*) malloc( sizeof( char ) * ( strlen( src ) + 1 ) );
	strcpy( newStr, src );

	return newStr;
}

void GUtil::getToday( char *dest )
{
	struct timeb timebuffer;
	struct tm *now;
	time_t ltime;
	
	ftime(&timebuffer);
	ltime = timebuffer.time;
	now = localtime(&ltime);

	sprintf( dest, "%04d_%02d_%02d-%02d_%02d_%02d",
		( 1900 + now->tm_year), ( now->tm_mon + 1 ), now->tm_mday, 
		now->tm_hour, now->tm_min, now->tm_sec );
}