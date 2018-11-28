package info.benjaminhill.bestsmiles

import mu.KotlinLogging
import net.tzolov.cv.mtcnn.FaceAnnotation
import net.tzolov.cv.mtcnn.MtcnnService
import java.awt.Point
import java.awt.Rectangle
import java.awt.image.BufferedImage
import java.io.File
import javax.imageio.ImageIO

private val logger = KotlinLogging.logger {}

fun FaceAnnotation.offset(loc: Point): FaceAnnotation {
    val newFa = FaceAnnotation()
    newFa.boundingBox = FaceAnnotation.BoundingBox.of(boundingBox.x + loc.x, boundingBox.y + loc.y, boundingBox.w, boundingBox.h)
    newFa.confidence = confidence
    newFa.landmarks = landmarks.map { FaceAnnotation.Landmark.of(it.type, FaceAnnotation.Landmark.Position.of(it.position.x + loc.x, it.position.y + loc.y)) }.toTypedArray()
    return newFa
}

fun FaceAnnotation.BoundingBox.toRectangle() = Rectangle(x, y, w, h)

fun FaceAnnotation.BoundingBox.intersects(other: FaceAnnotation.BoundingBox): Boolean = this.toRectangle().intersects(other.toRectangle())

fun getFaces(image: BufferedImage, mtcnnService: MtcnnService = MtcnnService()): List<FaceAnnotation> {
    val bgr = image.to3ByteBGR()
    val tileMap = bgr.toOverlappingTiles()
    logger.debug { "Created ${tileMap.size} overlapping tiles." }
    val overlappingFaces = tileMap.flatMap { (location, image) ->
        mtcnnService.faceDetection(image)!!.filterNotNull().also {
            logger.debug { "${location.x}x${location.y} = ${it.size}" }
        }.map {
            it.offset(location)
        }
    }

    val nonOverlappingFaces = mutableListOf<FaceAnnotation>()
    // Filter overlaps
    overlappingFaces
            .sortedByDescending { it.confidence }
            .filterTo(nonOverlappingFaces) { nextFa ->
                nonOverlappingFaces.firstOrNull { existingFa ->
                    existingFa.boundingBox.intersects(nextFa.boundingBox)
                } == null
            }
    return nonOverlappingFaces
}

fun BufferedImage.copyImage(): BufferedImage = BufferedImage(width, height, type).apply {
    createGraphics()!!.let { g2d ->
        g2d.drawImage(this, 0, 0, null)
        g2d.dispose()
    }
}

/** Not a huge memory drain because subimage uses same backing data */
fun BufferedImage.toOverlappingTiles(res: Int = 200): Map<Point, BufferedImage> {
    return (0..(height - res) step res / 2).flatMap { y ->
        (0..(width - res) step res / 2).map { x ->
            Point(x, y)
        }
    }.map { point ->
        point to getSubimage(point.x, point.y, res, res)
    }.toMap()
}

// Use MtcnnUtil in next version
fun BufferedImage.to3ByteBGR(): BufferedImage = if (type == BufferedImage.TYPE_3BYTE_BGR) {
    this
} else {
    BufferedImage(width, this.height, BufferedImage.TYPE_3BYTE_BGR).apply {
        createGraphics()!!.let { g2d ->
            g2d.drawImage(this, 0, 0, null)
            g2d.dispose()
        }
    }
}

fun main() {
    logger.debug { "Java: ${System.getProperty("java.version")}" }
    logger.debug { "OS: ${System.getProperty("os.name")} ${System.getProperty("os.version")} ${System.getProperty("os.arch")}" }
    logger.debug { "Hardware: Cores:${Runtime.getRuntime().availableProcessors()}, RAM(gb):${Runtime.getRuntime().freeMemory() / 1_073_741_824.0}" }

    val mtcnnService = MtcnnService()
    val imageFiles = File(System.getProperty("user.home"), "/Desktop/bestsmiles_in").walk().filter { it.extension.toLowerCase() == "jpg" }.toList().sorted()
    println("Total files: ${imageFiles.size}")

    imageFiles.forEach { file ->
        println("${file.name}\t${getFaces(ImageIO.read(file).to3ByteBGR(), mtcnnService).size}")
    }
}

