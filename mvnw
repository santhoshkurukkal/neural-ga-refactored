#!/bin/sh
#
# Maven Wrapper startup script
#

# Resolve links - $0 may be a softlink
PRG="$0"
while [ -h "$PRG" ] ; do
  ls=`ls -ld "$PRG"`
  link=`expr "$ls" : '.*-> \(.*\)$'`
  if expr "$link" : '/.*' > /dev/null; then
    PRG="$link"
  else
    PRG="`dirname "$PRG"`/$link"
  fi
done

# Get standard environment variables
if [ -z "$MAVEN_HOME" ] && [ -z "$M2_HOME" ]; then
  MAVEN_HOME="`dirname "$PRG"`/../.."
fi

# Java command
if [ -n "$JAVA_HOME" ] ; then
  JAVACMD="$JAVA_HOME/bin/java"
else
  JAVACMD="java"
fi

# Maven wrapper jar
WRAPPER_JAR="`dirname "$PRG"`/wrapper/maven-wrapper.jar"

# Check if wrapper jar exists, if not download it
if [ ! -f "$WRAPPER_JAR" ] ; then
  echo "Downloading Maven wrapper..."
  WRAPPER_URL="https://repo.maven.apache.org/maven2/org/apache/maven/wrapper/maven-wrapper/3.2.0/maven-wrapper-3.2.0.jar"
  if command -v curl > /dev/null; then
    curl -o "$WRAPPER_JAR" "$WRAPPER_URL"
  elif command -v wget > /dev/null; then
    wget -O "$WRAPPER_JAR" "$WRAPPER_URL"
  else
    echo "ERROR: curl or wget required to download Maven wrapper"
    exit 1
  fi
fi

# Run Maven wrapper
exec "$JAVACMD" \
  -Dmaven.multiModuleProjectDirectory="`dirname "$PRG"`/../.." \
  -jar "$WRAPPER_JAR" \
  "$@"